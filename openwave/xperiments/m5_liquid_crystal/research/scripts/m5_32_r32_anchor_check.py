import json, numpy as np
from scipy.optimize import brentq
me,mm,mt,smt=0.51099895,105.6583755,1776.93,0.09
out={}
def koide_tau(a,b):  # largest root of sum x = 2/3 (sum sqrt x)^2 for x=m, given a,b
    sa,sb=np.sqrt(a),np.sqrt(b)
    # s=sqrt(m_tau): a+b+s^2 = 2/3 (sa+sb+s)^2 -> quadratic in s
    c=sa+sb; A=1-2/3; B=-4/3*c; C=a+b-2/3*c*c
    return ((-B+np.sqrt(B*B-4*A*C))/(2*A))**2
# 1
t1=koide_tau(me,mm); out['1']=dict(tau=t1,pull=(t1-mt)/smt)
# 2
b=1e4; t=np.array([0.0071484,0.10279,0.42154])
tt=koide_tau(t[0]**2,t[1]**2)**.5   # Koide on t (m ~ t^2)
m_bt=b*t**2
tau_bt=koide_tau(m_bt[0],m_bt[1])
# eq 22 exact: yhat=m/b, s=(m/M)^2 -> t^2
M=3e7
def t2(m):
    y=m/b; s=(m/M)**2; return y+s*(1-y)**2/(1+s*(1-y))
te2,tm2=t2(me),t2(mm)
tt2=koide_tau(te2,tm2)  # Koide on t^2 amplitudes derived from PDG e,mu
tau_eq22=b*tt2
out['2']=dict(t_tau_from_koide_on_printed_t=tt,t_tau_printed=t[2],m_from_bt2=m_bt.tolist(),
  tau_koide_on_bt2_printed_t=tau_bt, tau_koide_on_t_pdg_e_mu_eq22=tau_eq22,
  tau_from_printed_ttau=m_bt[2], te_t_eq22=[np.sqrt(te2),np.sqrt(tm2)],
  tau_plain_koide_pdg=t1)
# 3
delta=lambda t:(b/M)**2*(t*t/(1-t*t))/(1+t*t/(1-t*t))**3
tev=t/np.sqrt(1)  # t=sin th ; r=tan th = t/sqrt(1-t^2)
r=t/np.sqrt(1-t**2); d=(b/M)**2*r**2/(1+r**2)**3
out['3']=dict(r=r.tolist(),f_r_TeV=(300*r).tolist(),delta=d.tolist(),kappa_h_minus1=(-2*d).tolist())
# W: try -(delta_mu-delta_e)*c; calibrate c on baseline (b=200,M=10TeV; t from eq22 with PDG masses)
def dl(m,b_,M_):
    y=m/b_; s=(m/M_)**2; t2_=y+s*(1-y)**2/(1+s*(1-y)); r2=t2_/(1-t2_)
    return (b_/M_)**2*r2/(1+r2)**3
base=dl(mm,200,1e4)-dl(me,200,1e4)
lb=dl(mm,10,3e4)-dl(me,10,3e4)
# LB1 delta using mass-derived t
out['3']['W_test']=dict(base_dmu_minus_de=base,base_stated_W=-0.105e-6,ratio_W_over_ddelta=-0.105e-6/base,
  lb1_dmu_minus_de_from_t=d[1]-d[0], W_if_minus_ddelta=-(d[1]-d[0]),W_if_minus_ddelta_over_2=-(d[1]-d[0])/2,stated=-5.7e-10)
# 4
alpha=1/137.035999; e=np.sqrt(4*np.pi*alpha); GF=1.1663788e-5; m=mm/1e3
Gmu=GF**2*m**5/(192*np.pi**3)
dcf=lambda M:e*m/(16*np.pi**2*M**2)
def BR(dL,dR): return alpha*m**3*(dL**2+dR**2)/4/Gmu
dmax=np.sqrt(4*1.5e-13*Gmu/alpha/m**3)  # if |dL|^2+|dR|^2=d^2
dmax2=dmax/np.sqrt(2)
d10_print=4.65e-13
dL=lambda M:d10_print*(1e4/M)**2*1.0
res={}
res['Gamma_mu_GeV']=Gmu
res['d_closed_form_10TeV']=dcf(1e4)
res['d_printed_10TeV']=d10_print
res['ratio_printed_over_closed']=d10_print/dcf(1e4)
res['d_bound_printed_L13']=1.445e-13
res['d_bound_from_BR1.5e-13_single_chirality_d']=dmax
res['d_bound_from_BR_if_both_chiralities_each_d']=dmax2
for tag,d1 in (('printed',d10_print),('closed',dcf(1e4))):
    d30=d1*(1e4/3e4)**2; deff=np.sqrt(2)*d30
    brv=BR(d30,d30)
    # BR with the BR-derived bound for d
    res[tag]=dict(d30=d30,deff_sqrt2=deff,margin_d_vs_L13=1.445e-13/deff,
       margin_d_vs_BR_bound=dmax/deff,BR_30TeV=brv,margin_rate_BR=1.5e-13/brv,
       margin_rate_from_d_sq=(1.445e-13/deff)**2,
       Mmin_now_from_L13=1e4*np.sqrt(np.sqrt(2)*d1/1.445e-13),
       Mmin_design_from_L13=1e4*np.sqrt(np.sqrt(2)*d1/(1.445e-13*np.sqrt(6e-14/1.5e-13))),
       Mmin_now_from_BR=1e4*np.sqrt(np.sqrt(2)*d1/dmax),
       Mmin_design_from_BR=1e4*np.sqrt(np.sqrt(2)*d1/(dmax*np.sqrt(0.4))),
       margin_rate_design_BR=6e-14/brv)
out['4']=res
json.dump(out,open('data/m5_32_r32_anchor_check.json','w'),indent=1,default=float)
print(json.dumps(out,indent=1,default=float))
