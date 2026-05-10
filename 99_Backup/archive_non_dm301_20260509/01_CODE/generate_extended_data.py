"""
generate_extended_data.py
=========================
QLD와 TQQQ의 가격 데이터를 1999년까지 소급 생성.

원리
----
레버리지 ETF는 기초 지수의 '일일 수익률'의 L배를 추종:
  r_gen(t) = α × r_QQQ(t) + β

  α : 유효 레버리지 (QLD≈2, TQQQ≈3)
  β : 일일 drag  (운용보수 + 차입비용, 음수)

교정 (Calibration)
------------------
OLS 회귀로 α, β 추정 — 최소제곱(MSE)으로 실제 가격 재현 최적화.
  r_real(t) = α × r_QQQ(t) + β + ε

교정 구간
  QLD  : 2006-06-21 ~ 현재  (약 19년 전체)
  TQQQ : 2010-02-11 ~ 현재  (약 16년 전체)

외삽 (Extrapolation)
--------------------
1. 교정된 (α, β)로 QQQ 전체 구간(1999~)에 synthetic 가격 생성
2. 상장일 기준으로 실제 가격에 맞게 스케일 조정 (연속성 보장)
3. 결합 시리즈  = synthetic(상장 전) + real(상장 후)
4. 저장 경로   : 02_DATA/yahoo_extended/{ticker}/{ticker}_daily.csv

출력
----
  - 02_DATA/yahoo_extended/QLD/QLD_daily.csv   (1999~)
  - 02_DATA/yahoo_extended/TQQQ/TQQQ_daily.csv (1999~)
  - 02_DATA/yahoo_extended/QQQ/QQQ_daily.csv   (원본 복사)
  - 03_RESULT/sensitivity/extended_data_validation.png
"""

import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "01_CODE")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Apple SD Gothic Neo"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATA_DIR  = Path("02_DATA/yahoo")
OUT_DIR   = Path("02_DATA/yahoo_extended")
PLOT_DIR  = Path("03_RESULT/sensitivity")

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_daily(ticker: str, base_dir: Path = DATA_DIR) -> pd.DataFrame:
    csv = base_dir / ticker / f"{ticker}_daily.csv"
    df  = pd.read_csv(csv, index_col="Date", parse_dates=True)
    return df[["Close"]]

print("데이터 로드...")
qqq  = load_daily("QQQ");  print(f"  QQQ  {qqq.index[0].date()} ~ {qqq.index[-1].date()}  ({len(qqq):,}일)")
qld  = load_daily("QLD");  print(f"  QLD  {qld.index[0].date()} ~ {qld.index[-1].date()}  ({len(qld):,}일)")
tqqq = load_daily("TQQQ"); print(f"  TQQQ {tqqq.index[0].date()} ~ {tqqq.index[-1].date()} ({len(tqqq):,}일)")

# ── 교정 (OLS on log-returns) ─────────────────────────────────────────────────
def calibrate(target: pd.DataFrame, qqq_all: pd.DataFrame, name: str):
    """OLS: r_target = α × r_qqq + β.  반환: (α, β, r²)"""
    common  = target.index.intersection(qqq_all.index)
    r_t     = np.log(target["Close"].loc[common] / target["Close"].loc[common].shift(1)).dropna()
    r_q     = np.log(qqq_all["Close"].loc[common] / qqq_all["Close"].loc[common].shift(1))
    r_q     = r_q.loc[r_t.index]  # align

    # OLS: [r_q, 1] × [α, β]' = r_t
    X       = np.column_stack([r_q.values, np.ones(len(r_q))])
    coef, _, _, _ = np.linalg.lstsq(X, r_t.values, rcond=None)
    alpha, beta   = coef

    # R² 계산
    r_pred  = alpha * r_q.values + beta
    ss_res  = np.sum((r_t.values - r_pred) ** 2)
    ss_tot  = np.sum((r_t.values - r_t.mean()) ** 2)
    r2      = 1.0 - ss_res / ss_tot

    # 연환산 drag
    annual_drag = beta * 252 * 100
    print(f"\n  [{name}] 교정 결과")
    print(f"    α (유효 레버리지) = {alpha:.6f}  (이론값 {round(alpha):d}×)")
    print(f"    β (일일 drag)    = {beta*1e4:.4f} bp/일  ({annual_drag:.2f}%/년)")
    print(f"    R²                = {r2:.6f}  ({r2*100:.3f}%)")
    return float(alpha), float(beta), r2

print("\n교정 중...")
a_qld,  b_qld,  r2_qld  = calibrate(qld,  qqq, "QLD")
a_tqqq, b_tqqq, r2_tqqq = calibrate(tqqq, qqq, "TQQQ")

# ── Synthetic 가격 시리즈 생성 ─────────────────────────────────────────────────
def build_synthetic(qqq_all: pd.DataFrame, alpha: float, beta: float,
                    real: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    1. QQQ 전체 구간에서 synthetic 생성 (시작값=100)
    2. 상장일에서 실제 가격에 맞도록 스케일
    3. 상장일 이후는 실제 데이터로 교체
    결과: Close 컬럼, 1999-03-10 ~ 현재
    """
    r_qqq  = np.log(qqq_all["Close"] / qqq_all["Close"].shift(1))
    r_syn  = alpha * r_qqq + beta

    # 가격 시리즈 (로그 누적합으로 구성)
    syn_price = 100.0 * np.exp(r_syn.fillna(0).cumsum())
    syn_price.iloc[0] = 100.0  # 첫날 초기화

    inception  = real.index[0]
    # 상장일 직전 거래일 기준으로 스케일
    prior_syn  = syn_price.loc[:inception]
    if inception in syn_price.index:
        scale = real["Close"].iloc[0] / syn_price.loc[inception]
    else:
        scale = real["Close"].iloc[0] / prior_syn.iloc[-1]

    syn_price_scaled = syn_price * scale

    # 결합: 상장 전 = synthetic(scaled), 상장 후 = real
    # 상장일 포함 이후는 real 사용
    pre_inception = syn_price_scaled[syn_price_scaled.index < inception]
    combined = pd.concat([pre_inception, real["Close"]]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    df_out   = pd.DataFrame({"Close": combined})
    df_out.index.name = "Date"

    # 통계
    print(f"\n  [{name}] 결합 시리즈")
    print(f"    전체 기간: {df_out.index[0].date()} ~ {df_out.index[-1].date()}  ({len(df_out):,}일)")
    print(f"    소급 기간: {df_out.index[0].date()} ~ {real.index[0].date() - pd.Timedelta(1, 'd')}  ({len(pre_inception):,}일 synthetic)")
    print(f"    연결점 확인: syn={syn_price_scaled.loc[syn_price_scaled.index < inception].iloc[-1]:.4f}  real_start={real['Close'].iloc[0]:.4f}")

    return df_out

print("\nSynthetic 시리즈 생성...")
qld_ext  = build_synthetic(qqq, a_qld,  b_qld,  qld,  "QLD")
tqqq_ext = build_synthetic(qqq, a_tqqq, b_tqqq, tqqq, "TQQQ")

# ── 저장 ─────────────────────────────────────────────────────────────────────
def save_extended(df: pd.DataFrame, ticker: str):
    dir_out = OUT_DIR / ticker
    dir_out.mkdir(parents=True, exist_ok=True)
    path = dir_out / f"{ticker}_daily.csv"
    df.to_csv(path)
    print(f"  저장: {path}  ({len(df):,}행)")

print("\n저장 중...")
save_extended(qld_ext,  "QLD")
save_extended(tqqq_ext, "TQQQ")
# QQQ 원본도 복사
qqq_out_dir = OUT_DIR / "QQQ"
qqq_out_dir.mkdir(parents=True, exist_ok=True)
qqq.to_csv(qqq_out_dir / "QQQ_daily.csv")
print(f"  저장: {qqq_out_dir / 'QQQ_daily.csv'}  ({len(qqq):,}행)")

# ── 검증 시각화 ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("레버리지 ETF 데이터 소급 생성 — 교정 및 검증\n"
             f"QLD (α={a_qld:.4f}×, β={b_qld*252*100:.2f}%/년)  |  "
             f"TQQQ (α={a_tqqq:.4f}×, β={b_tqqq*252*100:.2f}%/년)",
             fontsize=11, fontweight="bold")

# ① QLD 교정 적합도: 일일 수익률 산점도
ax = fig.add_subplot(gs[0, 0])
common_qld = qld.index.intersection(qqq.index)
r_qqq_c    = np.log(qqq["Close"].loc[common_qld] / qqq["Close"].loc[common_qld].shift(1)).dropna()
r_qld_c    = np.log(qld["Close"].loc[common_qld] / qld["Close"].loc[common_qld].shift(1)).loc[r_qqq_c.index]
r_fit_qld  = a_qld * r_qqq_c + b_qld
ax.scatter(r_qqq_c*100, r_qld_c*100, s=3, alpha=0.2, color="#9CA3AF")
x_line = np.linspace(r_qqq_c.min()*100, r_qqq_c.max()*100, 100)
ax.plot(x_line, a_qld*x_line + b_qld*100, color="#DC2626", lw=2, label=f"α={a_qld:.4f} β={b_qld*100:.5f}%")
ax.set_xlabel("QQQ 일일수익률 (%)", fontsize=9); ax.set_ylabel("QLD 일일수익률 (%)", fontsize=9)
ax.set_title(f"QLD 교정 (R²={r2_qld:.5f})", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ② TQQQ 교정 적합도
ax = fig.add_subplot(gs[0, 1])
common_tqqq = tqqq.index.intersection(qqq.index)
r_qqq_t     = np.log(qqq["Close"].loc[common_tqqq] / qqq["Close"].loc[common_tqqq].shift(1)).dropna()
r_tqqq_c    = np.log(tqqq["Close"].loc[common_tqqq] / tqqq["Close"].loc[common_tqqq].shift(1)).loc[r_qqq_t.index]
ax.scatter(r_qqq_t*100, r_tqqq_c*100, s=3, alpha=0.2, color="#9CA3AF")
x_line = np.linspace(r_qqq_t.min()*100, r_qqq_t.max()*100, 100)
ax.plot(x_line, a_tqqq*x_line + b_tqqq*100, color="#2563EB", lw=2, label=f"α={a_tqqq:.4f} β={b_tqqq*100:.5f}%")
ax.set_xlabel("QQQ 일일수익률 (%)", fontsize=9); ax.set_ylabel("TQQQ 일일수익률 (%)", fontsize=9)
ax.set_title(f"TQQQ 교정 (R²={r2_tqqq:.5f})", fontsize=9, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ③ QLD 누적 수익률 — 교정 구간에서 synthetic vs real 비교
ax = fig.add_subplot(gs[1, :])
# synthetic (교정구간만)
syn_r_qld    = a_qld * r_qqq_c + b_qld
syn_cum_qld  = (1 + syn_r_qld).cumprod()
real_cum_qld = (1 + r_qld_c).cumprod()
ax.plot(syn_cum_qld.index,  syn_cum_qld.values,  color="#DC2626", lw=1.5, alpha=0.9, label="Synthetic QLD (교정)")
ax.plot(real_cum_qld.index, real_cum_qld.values, color="#16A34A", lw=1.5, alpha=0.9, label="Real QLD")
ax.axvline(x=pd.Timestamp("2010-02-11"), color="#9CA3AF", ls=":", lw=1.5, label="TQQQ 상장일")
ax.set_title("QLD 교정 구간 검증: Synthetic vs Real 누적수익률", fontsize=9, fontweight="bold")
ax.set_ylabel("누적 수익률 (시작=1)", fontsize=9)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ④ 소급된 전체 시리즈 (1999~)
ax = fig.add_subplot(gs[2, :])
# 정규화: 1999-03-10=100
def normalize(s, base_date=None):
    if base_date is None: base_date = s.index[0]
    return s / s.loc[base_date] * 100

base = qqq.index[0]
qqq_n   = normalize(qqq["Close"], base)
qld_n   = normalize(qld_ext["Close"], base)
tqqq_n  = normalize(tqqq_ext["Close"], base)

ax.plot(qqq_n.index,   qqq_n.values,   color="#9CA3AF", lw=1.2, label="QQQ (실제)")
ax.plot(qld_n.index,   qld_n.values,   color="#DC2626", lw=1.2, label="QLD (synthetic+real)")
ax.plot(tqqq_n.index,  tqqq_n.values,  color="#2563EB", lw=1.2, label="TQQQ (synthetic+real)")
ax.axvline(x=qld.index[0],  color="#DC2626", ls="--", lw=1.2, alpha=0.7, label=f"QLD 상장 ({qld.index[0].date()})")
ax.axvline(x=tqqq.index[0], color="#2563EB", ls="--", lw=1.2, alpha=0.7, label=f"TQQQ 상장 ({tqqq.index[0].date()})")
ax.set_yscale("log")
ax.set_title("1999년부터 소급된 전체 시리즈 (로그 스케일, 1999-03-10=100)", fontsize=9, fontweight="bold")
ax.set_ylabel("정규화 가격 (로그)", fontsize=9)
ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)

plt.savefig(PLOT_DIR / "extended_data_validation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nextended_data_validation.png 저장 완료")

# ── 주요 이벤트 검증 ─────────────────────────────────────────────────────────
print("\n주요 이벤트 기간 QLD/TQQQ 최대 낙폭 검증:")
events = [
    ("닷컴 버블", "2000-01-01", "2002-12-31"),
    ("금융위기",  "2007-10-01", "2009-03-31"),
    ("COVID",    "2020-02-01", "2020-03-31"),
    ("2022 약세", "2021-12-01", "2022-12-31"),
]
for label, s, e in events:
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    for name, ser in [("QQQ", qqq["Close"]), ("QLD", qld_ext["Close"]), ("TQQQ", tqqq_ext["Close"])]:
        sub = ser.loc[(ser.index >= s) & (ser.index <= e)]
        if len(sub) == 0: continue
        peak = sub.cummax()
        dd   = ((sub - peak) / peak).min()
    print(f"  {label}:")
    for name, ser in [("QQQ", qqq["Close"]), ("QLD_ext", qld_ext["Close"]), ("TQQQ_ext", tqqq_ext["Close"])]:
        sub = ser.loc[(ser.index >= s) & (ser.index <= e)]
        if len(sub) == 0:
            print(f"    {name}: 데이터 없음")
            continue
        peak = sub.expanding().max().shift(1).fillna(sub.iloc[0])
        dd   = ((sub - peak) / peak).min() * 100
        print(f"    {name}: MDD={dd:.1f}%")

print("\n완료!")
