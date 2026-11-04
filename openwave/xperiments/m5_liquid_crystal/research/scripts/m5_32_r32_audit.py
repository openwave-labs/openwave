import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[v]="1"
import json, numpy as np
from scipy import ndimage
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data") + "/"
OUT=D+"m5_32_r32_audit.json"
n=24;h=1.5
c=(np.arange(n)-11.5)*h
X,Y,Z=np.meshgrid(c,c,c,indexing="ij"); R=np.sqrt(X**2+Y**2+Z**2)
seeds=["rad","obl","bia","perm","node"]
def tag(kind,s,J="J0"):
    if kind=="c0": return f"c0_{s}_J0_n24_L36_g8_d0.3_w25_c60_ls0_lsel0"
    if kind=="sx": return f"ax_sx_8.2_{s}_{J}_n24_L36_g8_d0.3_w25_c60_ls0_lsel0_sx8.2"
    return f"ax_sel_19.3_{s}_{J}_n24_L36_g8_d0.3_w25_c60_ls0_lsel0_sel19.3"
def load(t):
    M=np.load(D+"m5_32_r31_s1/"+t+".npz")["M"]; S=M[...,1:,1:]
    S=0.5*(S+np.swapaxes(S,-1,-2))
    return S,np.linalg.eigvalsh(S)
def gap(ev): return ev[...,1]-ev[...,0]
K=np.ones((3,3,3));K[1,1,1]=0
def strands(m,th):
    B=m>th
    cnt=ndimage.convolve(B.astype(int),K,mode="constant",cval=0)
    Zc=(m<th)&(R<9)&(cnt>=13)
    lab,nc=ndimage.label(Zc,structure=np.ones((3,3,3)))
    sizes=[int((lab==i).sum()) for i in range(1,nc+1)]
    return Zc,sizes
def links(S,ev):
    res=[]
    for ax in range(3):
        sl0=[slice(None)]*3; sl1=[slice(None)]*3
        sl0[ax]=slice(0,n-1); sl1[ax]=slice(1,n)
        a=tuple(sl0);b=tuple(sl1)
        dS=np.linalg.norm((S[b]-S[a]).reshape(*S[a].shape[:3],9),axis=-1)
        dl=np.linalg.norm(ev[b]-ev[a],axis=-1)
        ok=(R[a]<14)&(R[b]<14)
        jump=ok&(dS>0.3)&(dl<0.5*dS)
        idx=np.indices((n,n,n))[(slice(None),)+a]
        res.append((ok,jump,a,b))
    return res
def jumpstats(S,ev,m):
    tot=0;nj=0;inb=0;rads=[];near=0;jcells=np.zeros((n,n,n),bool)
    L=links(S,ev)
    for ok,jump,a,b in L:
        tot+=ok.sum();nj+=jump.sum()
        mm=np.zeros((n,n,n),bool);mm[a]=jump  # lower end marks
        jcells|=mm
        ma=m[a][jump]; inb+=(ma>0.03).sum()
        rads.append(np.minimum(R[a][jump],R[b][jump]) if False else 0.5*(R[a][jump]+R[b][jump]))
    rads=np.concatenate(rads) if rads else np.array([])
    hist=np.histogram(rads,bins=np.arange(0,14+1.5,1.5))[0].tolist()
    return dict(links=int(tot),jumps=int(nj),frac=nj/tot,inbiax_lower_end=(inb/nj if nj else None),radial_hist_w1p5_midpoint=hist),L
def near_strands(L,Zc):
    d=ndimage.binary_dilation(Zc,structure=np.ones((3,3,3)))
    j=0;t=0
    for ok,jump,a,b in L:
        j+=(jump&(d[a]|d[b])).sum();t+=jump.sum()
    return j/t if t else None
def bfrac(m,th,rmax=12): return float((m[R<rmax]>th).mean())
out={}
for kind in ["c0","sx","sel"]:
    for s in seeds:
        S,ev=load(tag(kind,s));m=gap(ev)
        r={}
        r["bfrac_r12"]={str(t):bfrac(m,t) for t in (0.01,0.03,0.1)}
        r["m_max"]=float(m.max());r["r_at_mmax"]=float(R.flat[m.argmax()])
        r["m_max_r12"]=float(m[R<12].max())
        sh=np.arange(0,16.5,1.5)
        r["shell_mean_m"]={f"{sh[i]:.1f}-{sh[i+1]:.1f}":float(m[(R>=sh[i])&(R<sh[i+1])].mean()) for i in range(len(sh)-1) if ((R>=sh[i])&(R<sh[i+1])).any()}
        r["bfrac_by_r"]={f"{x:.1f}":float((m[(R>=x)&(R<x+1.5)]>0.03).mean()) for x in np.arange(0,12,1.5)}
        Zc,sz=strands(m,0.03);r["strand_cells"]=int(Zc.sum());r["components"]=sorted(sz,reverse=True)
        js,L=jumpstats(S,ev,m);r["jump"]=js
        r["jump_within1_of_own_strand"]=near_strands(L,Zc)
        out[f"{kind}_{s}_J0"]=r
        if kind=="c0": cen=(Zc,m)
        if kind!="c0":
            Zcen=strands(gap(load(tag("c0",s))[1]),0.03)[0]
            vals=m[Zcen]
            r["m_on_center_loci"]=dict(n=int(vals.size),median=float(np.median(vals)) if vals.size else None,max=float(vals.max()) if vals.size else None,n_above_0p03=int((vals>0.03).sum()))
        if kind=="sx":
            # word logic
            c=gap(load(tag("c0",s))[1]);cf=bfrac(c,0.03)
            r["word"]=dict(ratio=r["bfrac_r12"]["0.03"]/cf,cond1=r["bfrac_r12"]["0.03"]<0.1*cf,cond2=bool((m[R<12]<0.03).all()),all_cells_below_0p1=bool((m[R<12]<0.1).all()))
for s in seeds:
    S,ev=load(tag("sx",s,"J1241.91"));m=gap(ev)
    r=dict(bfrac_r12={str(t):bfrac(m,t) for t in (0.01,0.03,0.1)},m_max=float(m.max()))
    r["jump"]=jumpstats(S,ev,m)[0];out[f"sx_{s}_J1241.91"]=r
json.dump(out,open(OUT,"w"),indent=1)
for k,v in out.items():
    print(k,{x:(v[x] if not isinstance(v[x],dict) or x in("bfrac_r12","word","m_on_center_loci") else '') for x in v if x in("bfrac_r12","m_max","strand_cells","components","word","m_on_center_loci","jump_within1_of_own_strand")},
      "jf=%.4f"%v["jump"]["frac"] if "jump" in v else "")
