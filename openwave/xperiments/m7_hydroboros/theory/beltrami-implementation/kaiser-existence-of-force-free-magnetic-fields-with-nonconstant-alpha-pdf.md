
Commun. Math. Phys. 211, 111 – 136 (2000) Communications in**Mathematical**
**Physics**© Springer-Verlag 2000
**On the Existence of Force-Free Magnetic Fields with Small****Nonconstantα****in Exterior Domains**
**R. Kaiser, M. Neudert, W. von Wahl**
Department of Mathematics, University of Bayreuth, 95440 Bayreuth, Germany
Received: 20 June 1999 / Accepted: 25 October 1999
**Abstract:**The existence of force-free magnetic fields in the exterior domain of some compact simply connected*surfaceS*is proved via an iteration scheme. The iteration starts with an arbitrary exterior vacuum field, which contains flux tubes originating and ending*onS.*At one cross-section of such a flux tube*withS*an arbitrary*functionα*is prescribed. For small values*ofα*(in the Hölder-norm*1,λ;*0*< λ <*1) the iteration is shown to converge to a force-free field with the prescribed values*ofα*in a flux tube which is close to the vacuum flux tube*andα*= 0 outside. The force-free field is close (in the Hölder- norm*1,λ)*to the starting vacuum field, in particular, it has the same field line topology, the same boundary values*onS*and satisfies the same decay conditions in spatial infinity. It is in general three-dimensional and requires no continuous symmetries.
**1. Introduction**
In the framework of a magnetohydrodynamic description of plasmas force-free magnetic fields play a prominent role. Their characteristic property is the alignment of magnetic field and electric current and thus the vanishing of the Lorentz-force. Magnetic*fieldB*
and current*densityj*then satisfy (in a suitable normalization) the equations
*j*=*α B,*
*curlB*=*j,*
*divB*=*0.*(1)
As a consequence of Eqs. (1) the scalar*functionα(x)*is constant on magnetic field lines but may vary from field line to field line. Force-free fields*withα*constant in space are a special case; for obvious reasons they are sometimes*termedlinear*force-free fields.
The interest in force-free fields has (at least) two sources. The first is “magnetic relaxation”: Suppose there is a viscous but perfectly conducting plasma together with a magnetic field contained in a*volumeV*. In general, the Lorentz-force does not vanish and
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQHsi2QOfbNVmhL0lWfu_WtPYb53auFK-jLTSoTzb_1QtxVDh158-yymku0ILJhBpJBwHbxN9iQLo0Rzy2UXgBtGtvI4Y8zyXFc2w9HGhH9Q2tj124P9U6SKFLkz_4K1szSgtoSY=w1146-h862-v0?authuser=0)

112 R. Kaiser, M. Neudert, W. von Wahl
**Fig. 1.A**magnetic flux tube in the solar corona
the plasma, even if initially at rest, is set into motion. If there are no external forces the total energy*inV*must decrease because of viscous dissipation. Eventually, the plasma comes to rest, i.e. the system relaxes to a state of magnetostatic equilibrium. During this process the topology of knots and links of the magnetic lines remains conserved since by virtue of the perfect-conductivity assumption magnetic lines are frozen in the plasma. As a consequence magnetic helicity in any flux tube is a conserved quantity. The relaxed state is thus obtained by minimizing the total energy with the constraint of conserved helicity. These minimizing solutions are always force-free magnetic fields ([Wol], [Tay]). If the constraint of conserved helicity in any flux tube is weakened to conservation of helicity in the total plasma volume the relaxed state turns out to be a linear force-free field. However, besides the obvious advantage of a simplified problem, theoretical rationalizations of this replacement are doubtful ([Gra]).
The other reasons for interest in force-free magnetic fields are astrophysical applications. Because of the high electrical conductivity of stellar material of even low density, stellar magnetic fields are usually accompanied by large electric currents ([LS], [Ch1]). If, moreover, the magnetic energy density dominates the plasma pressure which is the typical situation in stellar atmospheres, stationary magnetic fields have to be force-free. So, a variety of magnetic structures which have been observed in active regions of the solar corona (coronal loops, magnetic arcades, coronal mass ejections, etc.) have been modelled on the basis of force-free fields, see for instance [Pri] or [Bra]. Solar flares, for example, are nowadays attributed to a magnetic origin ([Al2]): Magnetic flux emerges from the solar convection zone into the coronal space. The simplest geometry is a looptype flux tube with footpoints anchored in the photosphere (see Fig. 1). Photospheric motions cause then a shearing and twisting of the magnetic flux tube up to the point where the flux tube becomes unstable. Finally, a substantial fraction of the free energy stored in the magnetic structure is released (probably by reconnection of magnetic field lines) in an eruptive event. Applications like these determine the geometric setting we will consider below.
For*linear*force-free fields a well-posed boundary value problem of Neumann’s type can be formulated. Using integral equations ([Kr1]) or Hilbert space methods ([Sak], [YG]) this problem has been solved under quite general conditions. In*theonlinear*case comparable results exist only if the problem is modified or restricted: A modified problem is obtained if the divergence-free condition is abandoned. In that case the abovementioned methods still work ([Pic], [Kr2]); these solutions, however, are obviously not appropriate for the description of magnetic fields and are merely of mathematical interest. The restricted problem refers to situations with plane – or axisymmetry. In that case the problem can be reduced to a single, in general nonlinear equation of elliptic
On the Existence of Force-Free Magnetic Fields 113
type for a scalar quantity describing the symmetric field ([LS]). For equations of such type an elaborate mathematical theory is available. Beyond these results there are yet a few special solutions developed for astrophysical purposes which are nonlinear and nonsymmetric ([CC], [Low]) and some nonexistence results in the whole space ([Ch2]) and in exterior domains ([Al1]) but no general existence results. What comes so far closest to a general solution has been obtained via an iteration scheme which works for*smallα*and special initial configurations. This scheme has been proposed by [GR] and is here*calledα-iteration*in analogy to*theβ-*iteration, which has been devised for the more general problem of the existence of magnetohydrostatic equilibria ([Spi], [Lor]).
*Then*th step of*theα-iteration*has the form
*〈Bn,∇αn〉*= 0 in*G, αn|M*=*α0 M*⊂*∂G,*(2)
*curlBn+1*=*αnBn*in*G, divBn+1*= 0 in*G, 〈ν, Bn+1〉*=*g*on*∂G.*
(3)
*Here,G*denotes an open domain inR3,*〈. , .〉*the euclidian scalar product inR3,*ν*is the exterior unit normal*on∂G,*the*functiong*is prescribed*on∂G*and the*functionα0*
on such a*partM*of*∂G, where〈ν, B0〉*6= 0. Each step of the iteration requires the solution of an initial value problem for a linear first order differential equation (2) and of a Neumann-boundary value problem for inhomogeneous-harmonic vector fields (3). In case of convergence the iteration furnishes obviously a force-free magnetic field.
In case of*aboundeddomainG*Bineau ([Bin]) proved convergence of the iteration for*smallα0*and a harmonic initial*fieldB0.*A nonzero lower bound on the field strength of*B0*ensures finite length of all field lines in the flux tube emanating*fromM.*As a result Bineau obtains a force-free magnetic field close*toB0*with the prescribed values of*α*in a flux tube close to the initial one emanating*fromM.*The proof of convergence, however, is at least incomplete. Bineau assumes a-priori bounds*on‖B‖ and‖∇B‖*as well as on the field line parameter without controlling these bounds in the course of the iteration. But in a rigorous treatment these bounds have to be controlled at every step of the iteration and, in particular, the bounds have to be uniform with respect to the iteration*numbern.*This is no trivial matter since the bounds on the magnetic field and on the field line parameter depend on each other.
The present paper furnishes also a convergence proof for*theα-iteration,*but differs in two respects from the work of Bineau. First, motivated from the astrophysical applications described above the underlying domain is not bounded but the exterior of a bounded simply connected domain. That is why we need some nonstandard potential-theoretic estimates in exterior domains (these are cited in Sect. 3) and why we have to formulate carefully the notion of an “admissible field configuration” (Sect. 4). The guideline is here again the astrophysical situation. Second, our convergence proof (Sect. 5) avoids the above mentioned shortcomings and presents in some detail the required estimates. One should note that the proof can easily be carried over to the simpler case of a bounded domain as well as to the case of the exterior of a bounded but multiply connected domain (see Remarks 3.3 and 5.6).
114 R. Kaiser, M. Neudert, W. von Wahl
**2. General Notations**
In the following, we*assumeG*⊂ R 3 to be a bounded domain with smooth boundary; by
“smooth” we mean sufficient regularity*of∂Gwithout*fixing the exact class of regularity. In all cases considered here the*classC6*will be sufficient.*Furthermore,Ĝ*means the exterior*ofG,*i.e.
*Ĝ*:= R 3*\G.*
*ν*denotes, if not explicitly defined otherwise, the outer normal with respect*toG.*A closed*surfaceS*is here the smooth*boundary∂D*of some bounded*domainD*⊂ R
3,*whereD*always lies locally at one side of the boundary. Then, in*particular,S*=*∂D*is orientable.
For*δ >*0 we define
*G−δ*:= {*x*∈*G∣∣*dist*(x, ∂G) > δ*
}*, ∂G−δ*:=*∂*
(*G−δ).*
The euclidian scalar product inR3 is denoted*by〈. , .〉,*i.e.
〈*x, y*
〉 := 3∑*j=1*
*xjyj .*
For some differentiable vector*fieldB,*the Jacobian matrix will be denoted*byDB.*For multiindices a=*(a1,a2,a3)*∈ N
3 0 we set
*Daf*:=*∂ |a|f ∂x*
a1 1*∂x*
a2 2*∂x*
a3 3*,*|a| := a1 + a2 +*a3,*a! :=*a1!a2!a3!.*
For a continuous function or vector*fieldf*in some (bounded or unbounded) domain*G*we*set‖f ‖C0(G)*:= sup
*x∈G |f (x)|.*Furthermore, for some function or vector*fieldf*
*andk*∈ N
‖*Dkf*‖ :=*max|a|=k*‖*Daf*‖*.*
If some*matrixA*=*(aij )*:*G*⊂ R 3 → R
3×3 is continuous, we set
‖*A ‖C0(G)*:= ( 3∑*i,j=1*
*‖aij‖2 C0(G)*
*)1/2 .*
**3. Hölder-Spaces and the Neumann-Problem for Inhomogeneous-Harmonic Vector Fields in Exterior Domains**
Hölder spaces are the appropriate function spaces in classical potential theory. the socalled*LetG*⊂ R
3 be a domain. For a function or a vector*fieldf*∈*Ck(G)and 0< λ <*1 we define then
‖*f ‖Ck,λ(G)*:=*k∑ j=0*
‖*Djf ‖C0(G) +[Dkf ]Cλ(G),*(4)
On the Existence of Force-Free Magnetic Fields 115
where
*[Dkf ]Cλ(G)*:=*max|a|=k*sup*x,y∈G*
*|Daf (x)−Daf (y)| |x*−*y|λ .*
Distinguishing between local and global Hölder continuity, the following notation is customary in the literature (see, for instance, [GT]): In the local case the*condition‖f ‖Ck,λ(K) <*∞ holds for every compact*subsetK*⊂*G andf*belongs therefore to the*spaceCk,λ(G)*:=*C*
*k,λ*loc*(G).*If, however, the uniform
condition
*‖f*‖*Ck,λ(G) <*∞
*holds,f*is said to be
*f*∈*Ck,λ(G)*:=*C k,λ unif(G)*
*andCk,λunif(G)*is a Banach space with the norm defined in (4). In order to simplify our
notation and because we are not interested in the*spacesC k,λ*loc*(G),*deviating from this
convention we set
*C1,λ(Ĝ)*:=*C 1,λ unif(Ĝ), Cλ(Ĝ)*:=*C*
*0,λ unif(Ĝ),*
*whereĜ*:= R 3*\G*for some bounded*domainG*⊂ R
3. For*everyf*∈*C1,λ(Ĝ)*there exists an*extensionf*∈*C1,λ(R3), f*
∣∣*Ĝ*
=*f*such that
*‖f ‖C1,λ(R3)*≤*c1 ‖f*‖*C1,λ(Ĝ)*
*, ‖f ‖C0(R3)*≤*c1 ‖f*‖*C0(Ĝ)*
(5)
with some*constantc1 >*0 independent*off*(see [GT], Lemma 6.37; the unboundedness of*Ĝ*is not relevant here). We note that this construction is linear.
Later on the following properties of the Hölder norm will turn out to be useful:
*[f*·*g]Cλ(G)*≤*‖f ‖C0(G)*·*[g]Cλ(G)*+*[f ]Cλ(G)*·*‖g‖C0(G),*(6)
*‖f*·*g‖Cλ(G)*≤*‖f ‖Cλ(G)*·*‖g‖Cλ(G),*(7)
*[1/f ]Cλ(G)*≤*‖1/f*‖2*C0(G)*·*[f ]Cλ(G).*(8)
Concerning the*boundary∂G ofG*(or*Ĝ) f*∈*C1,λ(∂G)meansf ◦µ*∈*C1,λ(U),*where*µ*:*U*→*∂G*is a chart*of∂G.*The corresponding*norm‖.‖C1,λ(∂G)*therefore depends on the chosen atlas.
For a*subsetM*⊂*∂G*which is open in the topology*of∂G*and*somef*∈*C1,λ(M)*
with compact support*inM*we also*writef*∈*C1,λ*0*(M).*
In addition to the Hölder-norm we also need in exterior domains a weighted norm characterizing the asymptotic behaviour to obtain global estimates for potential theoretic problems. Here for a function or a vector*fieldf*in*Ĝ and% >*0 we set
|‖*f |‖%:=*sup*x∈Ĝ*
*|x|%|f (x)|.*
If |‖*f |‖%<*∞ we also*write|f (x)|*=*O(|x|−%), |x|*→ ∞. The following potential theoretic results on the Neumann problem for inhomo-
geneous-harmonic vector fields in exterior domains are cited without proof. The proofs can be found in [NvW].
116 R. Kaiser, M. Neudert, W. von Wahl
**Theorem 3.1 (Solvability**and asymptotic**behaviour).***LetG*⊂ R 3*be a bounded do-*
*main with smooth boundary and trivial topology(i.e. first and second Betti number are zero), Ĝ*:= R
3*\G, 1< % < 3. Let, furthermore,*
*f*∈*Cλ(Ĝ,R) , |f (x)|*=*O*
*(|x|−%), |x|*→*∞,*
*w*∈*C1,λ(Ĝ,R3)*∩*Cλ(Ĝ,R3), w with zero flux,*
*|w(x)|*=*O (|x|−%), |x|*→*∞,*
*g*∈*C0(∂G,R).*
*Then the Neumann problem*
div*v*=*f, curlv*=*w in Ĝ, 〈v, ν〉*=*g on ∂G, |v(x)|*=*O*
*(|x|1−%), |x|*→ ∞*has a unique solution.*
*If f*=*0, w*= 0*then|v(x)|*=*O (|x|−2*
)*, |x|*→*∞.*
*If f*=*0, w*= 0*and*∫*∂G*
*g d*= 0*then even|v(x)|*=*O (|x|−3*
)*, |x|*→*∞.*
The condition*“w*with zero flux” is necessary for the solvability of the problem and means that for all closed oriented*surfacesS*⊂*Ĝ*with outer*normalν̃*the integral∫*S*
〈*w, ν̃*
〉*d*vanishes. This implies*divw*= 0.
**Theorem 3.2 (Hölder-Estimates).***Suppose that1 < % < 3, g*∈*C1,λ(∂G) and letv be the unique solution of the Neumann problem3.1. Thenv*∈*C1,λ(Ĝ) and there exists a constantc0 >*0*only depending onλ, %,G with*
‖*v*‖*C1,λ(Ĝ)*
+ |‖*v |‖%−1*
≤*c0*·*(‖f*‖*Cλ(Ĝ)*
+*‖w‖ Cλ(Ĝ)*
+ |‖*f |‖%*+ |‖*w |‖%*+ ‖*g ‖C1,λ(∂G)*
)*.*
*Remarks 3.3.(a)The*corresponding Neumann-problem in the interior domain is solvable under the additional restriction ∫
*G*
*f dx*= ∫*∂G*
*gd.*
The asymptotic conditions, of course, are to be omitted. (b) If the exterior*domainĜ*is multiply connected, i.e. if the first*Betti-numberñ*of*Ĝ,*which is the number of handles*ofG*or*Ĝ,*is different from zero, the Neumann-problem is still uniquely solvable under the same conditions, if the so-called generalized circulations
*0j*:= ∫*∂G*
*〈ν*×*v,***zj***〉d, j*=*1, . . . , ñ*
On the Existence of Force-Free Magnetic Fields 117
are prescribed,**where{z1,***. . . ,***zñ}**is an appropriate basis of the space
**ZR(G)**:=*{v*∈*C1(G,R3)*∩*C0(G,R3)|*div*v*=*0, curlv*= 0*inG, 〈ν, v〉*= 0*on∂G}.*
The asymptotic behaviour of the solution then is the same as stated in Theorem 3.1, but to*obtain|v(x)|*=*O(|x|−3), |x|*→ ∞, the*condition01*= · · · =*0ñ*= 0 is necessary. The estimate in Theorem 3.2 holds, if the term
*∑ñ j=0 |0j*| is added in the brackets on
the right-hand side.
**4. Field Lines and Admissible Configurations**
Field lines are here considered as curves with direction always parallel to the field vector and orientation which is induced by the vector field. Taking into account the invariance under reparametrization they can be described as solutions of the so-called field line equation
*γ̇ (t)*=*B*(*γ (t)*
)*,*
*whereB*stands for the vector*field,t*for the curve parameter and “· ” means differentiation with respect*tot*. In the following we are interested in field line configurations which contain flux tubes in an exterior*domainĜ*with footpoints anchored*on∂G.*
**Definition 4.1 (Assumptions**and further**notations).**
(1)*LetG*⊂ R 3*be a bounded domain with smooth boundary,Ĝ*:= R
3 \*G, U*⊂ R 2
*open,M*⊂*∂G open,*(*U,µ,µ(U)*=*M)*
*a local coordinate system forM, i.e. in particularµ*:*U*→*M is a homeomorphism andµ*:*U*→ R
3*is twice continuously differentiable. Letρ1 >*0*such that*
*∀s*=*(s(1), s(2))*∈*U*: ∣∣∣∣*∂µ∂s(1) (s)*×*∂µ*
*∂s(2) (s)*
∣∣∣∣ ≥*ρ1.*
(2)*LetB*∈*C1,λ(Ĝ,R3) andρ0 >*0*such that*〈*B, ν*
*〉∣∣M*≥*ρ0, i.e. in particular, the field lines ofB emanating fromM have non-zero normal component. Fors*∈*U let γ (., s) denote the solution of the initial value problem*
*γ̇*=*B(γ ) in Ĝ, γ (0, s)*=*µ(s)*∈*M.*
*Let ]0, T (s)[ be the maximum interval of existence ofγ (., s) in Ĝ, T (s)*∈ R+ ∪*{+∞}. We define now*
*0 (M, B*
) := {*(t, s)*∈ R
3*∣∣s*∈*U, 0< t < T (s)*
}*,*
*3 (M, B*
) := {*γ (t, s)*
*∣∣(t, s)*∈*0(M, B*)}*.*
*3 (M, B*
)*may be considered as the flux tube ofB in Ĝ emanating fromM. The*
*parameterst, s will be denoted as field line coordinates. For T >*0*we define furthermore*
*0 (M, B, T*
) := {*(t, s)*∈ R
3*∣∣s*∈*U, 0< t <*min
(*T , T (s)*
)}*,*
*3 (M, B, T*
) := {*γ (t, s)*
*∣∣(t, s)*∈*0(M, B, T*)}*.*
118 R. Kaiser, M. Neudert, W. von Wahl
Observe that the solutions*ofγ̇*=*B(γ )*are restrictions of the solutions*ofγ̇*=*B(γ ), sinceB*
∣∣*Ĝ*
=*B, whereB*denotes the extension*ofB*(see Sect.3). From the theory of ordinary differential equations we shall use the following results:
**Lemma****4.2.Let***G*⊂ R 1+3*be open,f*:*G*→ R
3*continuous,I*⊂ R*an interval with*0 ∈*I , ϕ1, ϕ2*∈*C1(I,R3) with*
*∀t*∈*I*:*(t, ϕi(t))*∈*G, i*=*1,2,*
*andε0, δ0, L > 0. Let, furthermore, the following conditions be satisfied:*
*(i)*|*ϕ1(0)− ϕ2(0)*| ≤*ε0,*
*(ii ) ∀t*∈*I*: |*ϕ̇i (t)− f (t, ϕi(t))*| ≤*δ0, i*=*1,2, (iii ) ∀t*∈*I*: |*f (t, ϕ1(t))− f (t, ϕ2(t))*| ≤*L*|*ϕ1(t)− ϕ2(t).*
*Then for allt*∈*I the estimate holds:*
|*ϕ1(t)− ϕ2(t)*| ≤ (*ε0*+*2δ0*|*t*| )
*eL|t .*
The following lemma describes the dependence of solutions of an ordinary differential on the initial values.
**Lemma****4.3.Let***G*⊂ R 1+3*be open ,f*:*G*→ R
3*continuously differentiable with respect tox*∈ R
3*. Letϕ(.,0, y) denote the solution of the initial value problem*
*ẋ*=*f (t, x), x(0)*=*y.*
*Then there exists*
*vj (t)*:=*∂ϕ*
*∂yj (t,0, y0), j*=*1,2,3,*
*for (0, y0)*∈*G, andvj solves the initial value problem*
*v̇j (t)*= (*Dxf*
)*(t, ϕ(t,0, y0))*·*vj (t), vj (0)*=*ej , j*=*1,2,3.*
Using these propositions and Definition 4.1 the following lemma is easy to prove.
**Lemma****4.4.3(M,***B) is open inĜ.*
**Definition 4.5 (Admissible****configuration).AssumeG,***Ĝ,U,µ,M,as in Definition4.1, ρ0, δ, T > 0, B*∈*C1,λ(Ĝ,R3) with extensionB*∈*C1,λ(R3,R3). We denote the pair(M, B*
)*as an admissible configuration with parameters*
(*ρ0, T , δ*
)*, iff the following*
*conditions are satisfied:*
(i) 〈*B, ν*
〉 ≥*ρ0 >*0*in M.*(ii)*For the solutionγ (., s) of the initial value problem*
*γ̇*=*B(γ ) in*R*3, γ (0, s)*=*µ(s)*∈*M,*
*we have*
∀*s*∈*U*: ∃*T0(s)*∈ ]0;*T2*[ :*γ*(*T0(s), s*
) ∈*∂G,*∃*Tδ(s)*∈ ]0;*T2*[ :*γ*
(*Tδ(s), s*
) ∈*∂G−δ.*
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQEFqcUnzRB5ldUYulNTvJBAxcYm3bozKta4cyDRwupwk8ABQhvLU8hJIo_t09LyANN9pLOF_OVOhzojF02UyzLeAIgotauOGBK_8pBiwiq3plRirr70nUZDkTB3PORFK9l-19V3eg=w1107-h576-v0?authuser=0)

On the Existence of Force-Free Magnetic Fields 119
**Fig. 2.Admissible**field configuration
The second condition in the definition above has the following meaning: Every field line of*B*in*Ĝ*starting*fromM*returns to the*surface∂G*with a finite value of*parametert*. There is a uniform upper*boundT2*for the*valuest*=*T0(s) andt*=*Tδ(s)where*the line*γ (t, s)*starting*inµ(s)*penetrates the*surfaces∂G and∂G−δ*(see Fig.2).
A non-vanishing penetration depth of the field lines is guaranteed in particular, if the absolute value of the normal component*ofB*on the corresponding part of the surface is bounded from below.
**Lemma****4.6.Suppose***that the configuration (M, B*
)*is admissible with parameters(*
*ρ0, T , δ*)*. Then*
*diam3(M, B) <*‖*B*‖*C0(Ĝ)*
·*T*+*diamM.*
*Proof.*We have
*γ (t, s)− γ (0, s)*=*t∫*
0
*γ̇ (τ, s) dτ*=*t∫*
0
*B*(*γ (τ, s)*
)*dτ,*
so ∣∣*γ (t1, s1)− γ (t2, s2)*∣∣ ≤ ∣∣*γ (t1, s1)− γ (0, s1)*
∣∣ + ∣∣*γ (0, s1)*−*γ (0, s2)*
∣∣ + ∣∣*γ (0, s2)− γ (t2, s2)*∣∣
≤ |*t1*| · ‖*B*‖*C0(Ĝ)*
*+diamM+*|*t2*| · ‖*B*‖*C0(Ĝ)*
*<*‖*B*‖*C0(Ĝ)*
·*T*+*diamM.*ut**Lemma****4.7.AssumeG,***Ĝ, µ,M, γ as in Definition4.1,*
*ρ0, T , δ > 0, B*∈*C1,λ(Ĝ,R3),*
*the configuration (M, B*
)*admissible with parameters*
(*ρ0, T , δ*
)*, η*∈*]0,1[, B*′ ∈
*C1,λ(Ĝ,R3) satisfying*
*(i)*〈*B ′, ν*
〉 = 〈*B, ν*
〉*in M and*
*(ii )*‖*B*′ −*B*‖*C0(Ĝ)*
*< (1*−*η)δ*
*c1T*· exp
{ − 1
2*c1*‖*B*‖
*C1(Ĝ)*·*T*}
*.*
*Then (M, B*′)*is admissible with parameters*
(*ρ0, T , ηδ*
)*.*
120 R. Kaiser, M. Neudert, W. von Wahl
*Proof. Letγ*denote the solution of the field line equation with respect*toB, γ*′ the same with respect*toB*′ according to Definition 4.1. Then, using the estimate (5) and Lemma 4.2*(withε0*= 0,*f*=*B, L*=*‖DB‖*
*C0(Ĝ)*) we get for 0≤*t*≤*T*
2 ,
|*γ ′(t, s)− γ (t, s)*| ≤ 2 ‖*B*′ −*B ‖C0(R3)*·*t*· exp{‖*DB ‖C0(R3) ·t}*≤*2c1*‖*B*′ −*B*‖
*C0(Ĝ) ·t*·*exp{c1*‖*B*‖
*C1(Ĝ)*·*t}.*
Thus, assumption (ii)*yields∣∣γ ′(Tδ(s), s)*−*γ*(*Tδ(s), s*
)∣∣ ≤*(1*−*η)δ.*
For*allx*∈*∂G*we have therefore∣∣*γ ′(Tδ(s), s)*−*x*∣∣
≥*∣∣x*−*γ*(*Tδ(s), s*
)∣∣ −*∣∣γ ′(Tδ(s), s)*−*γ*(*Tδ(s), s*
)∣∣ ≥*δ*−*(1*−*η)δ*=*ηδ.*
The lemma follows then by continuity*ofγ ′(. , s).*ut In the proof of the lemma to follow the field line coordinates will turn out to be useful to solve the first order linear partial differential*equation〈∇ψ,B〉*=*ϕ*
in flux tubes of the*fieldB.*
**Theorem 4.8 (Solvability**of the initial value**problem).***LetG, Ĝ, µ,M,3(M, B) be as in Definition4.1, ρ0 > 0, B*∈*C1(Ĝ,R3) with*‖*B*‖
*C1(Ĝ) <*∞*and*
〈*B, ν*
〉 ≥*ρ0 in M.*
*Furthermore, assume*
*ψ0*∈*C1(M,R) and ϕ*∈*C0(3(M, B),R*)*.*
*Then the initial value problem〈 ∇ψ,B*〉 =*ϕ, ψ ∣∣M*=*ψ0*(9)
*has a unique solution in3(M, B).*
*Proof.*(a)*Letγ (., s) andT (s)*be as in Definition 4.1.*Since|B|*is uniformly bounded, we have either
*T (s)*=*∞,*
which is here also admitted, or
lim*t↗T (s)*
*γ (t, s)*∈*∂G,*
in the*caseT (s) <*∞. From the uniqueness theorem for ordinary differential equations with Lipschitz-condition we know that every point lies on exactly one field line and that two different field lines cannot cross each other.
On the Existence of Force-Free Magnetic Fields 121
(b) We show next that a field line*ofB*in*Ĝ*starting*fromM*cannot return*toM.*Let
*ν̃(s)*:=*ν*(*µ(s)*
) be the outer normal with respect*toG*in the*pointµ(s)*∈*M*⊂*∂G.*
We consider some*solutionγ (., s0)*of the field line equation according to Defini-tion 4.1 and assume that there is*somet1 >*0, with
*γ (t1, s0)*:= lim*t↗t1*
*γ (t, s0)*∈*M*
*andγ (t, s0)*∈*Ĝ*for*t*∈*]0, t1[.*Moreover,*forτ >*0 sufficiently small, there exist continuous*functions(.)*:*]t1*−*τ, t1[→ U andϑ(.)*:*]t1*−*τ, t1[→*R such that
*γ (t, s0)*=*µ(s(t))+ ϑ(t)*·*ν̃(s(t)).*(10)
We may*assumes(.), ϑ(.)*to be differentiable, and (without restriction)
*ϑ̇(t) <*0*on]t1*−*τ, t1[*and*ϑ(t)*↘ 0 for*t*↗*t1.*
Differentiating Eq. (10) with respect*tot*yields
*γ̇ (t, s0)*=*ṡ(1)(t)*·*∂µ*
*∂s(1)*
(*s(t)*
) +*ṡ(2)(t)*·*∂µ*
*∂s(2)*
(*s(t)*
) +*ϑ̇(t)*·*ν̃(s(t))*
+*ϑ(t)*
(*ṡ(1)(t)*·*∂ν̃*
*∂s(1)*
(*s(t)*
) +*ṡ(2)(t)*·*∂ν̃*
*∂s(2)*
(*s(t)*
))*.*
(11)
Taking the scalar product of Eq. (11)*withν̃*(*s(t)*
) we get
〈*γ̇ (t, s0), ν̃*
(*s(t)*
)〉 =*ϑ̇(t) < 0,*
and thus 〈*B*
(*γ (t1, s0)*
)*, ν̃*
(*s(t1)*
) 〉 ≤*0,*
which contradicts our assumption.
(c) We first consider the initial value problem with homogeneous differential equation 〈*∇ψhom, B*
〉 =*0, ψhom ∣∣M*=*ψ0 .*(12)
Equation (12)1 means*thatψhom*is constant along the field lines*ofB:*
*∂*
*∂t*
(*ψhom(γ (t, s))*
) = 〈*∇ψhom*(*γ (t, s)*
)*, γ̇ (t, s)*
〉 = 〈*∇ψhom*
(*γ (t, s)*
)*, B*
(*γ (t, s)*
) 〉 =*0.*
Therefore any solution of problem (12) satisfies
*ψhom*(*γ (t, s)*
) =*ψhom*(*γ (0, s)*
) =*ψ0*(*µ(s)*
)*.*(13)
From (a) and (b) we know*thatγ (., .)*:*0(M, B*) →*3*
*(M, B*)
is a bijective mapping. Thus,*conversely,ψhom*:*3(M, B*
) → R defined by Eq. (13) solves the problem (12).
122 R. Kaiser, M. Neudert, W. von Wahl
*Thereforeψhom*is the unique solution of (12). For the inhomogeneous problem
〈*∇ψinh, B*〉 =*ϕ, ψinh*
*∣∣M*= 0
we can easily find the solution
*ψinh*(*γ (t, s)*
) =*t∫*
0
*ϕ*(*γ (t ′, s)*
)*dt*′
by differentiating both sides with respect*tot*. Finally,
*ψ*(*γ (t, s)*
) =*ψhom*(*γ (t, s)*
) +*ψinh*(*γ (t, s)*
)
=*ψ0*(*µ(s)*
) +*t∫*
0
*ϕ*(*γ (t ′, s)*
)*dt*′
is the unique implicitly given solution of the initial value problem (9).ut Considering the preceding proof and Lemma 4.3 the next lemma follows easily.
**Lemma****4.9.LetG,***Ĝ, µ,M, γ,3(M, B) be as in Definition4.1, ρ0, T , δ > 0, B*∈*C1,λ(Ĝ), α0*∈*C1*
*0(M,R) and the configuration(M, B) admissible with parameters (ρ0, T , δ). Furthermore, letα be the solution of the initial value problem*
〈*∇α,B*〉 =*0, α ∣∣M*=*α0*
*in 3(M, B). Then*
*suppα*∩*Ĝ*⊂*3 (M, B*
)*.*
*Thereforeα is trivially extendable toĜ.*
**Lemma****4.10.In***addition to the assumptions given in Lemma4.9 supposedivB*=*0. ThenαB has zero flux inĜ.*
*Proof.*Because of Lemma 4.9 we have
supp (*αB*
) ∩*Ĝ*⊂*3 (M, B*
)*,*
and Lemma 4.6 yields
*diam3 (M, B*
)*<‖ B*‖
*C0(Ĝ) ·T*+*diamM.*
For sufficiently*largeR >*0 we have thus
*G*∪*supp(αB)*⊂*KR(0).*
With*αB*∣∣*∂KR(0)*
= 0,
div (*αB*
) =*α divB*+*〈∇α,B〉*= 0*inKR(0) \G*
On the Existence of Force-Free Magnetic Fields 123
and applying Gauß’s theorem*toKR(0) \G*we obtain ∫*∂G*
〈*αB, ν*
〉*d*=
∫*∂KR(0)*
〈*αB, ν*
〉*d*−
∫
*KR(0)\G*div
(*αB*
)*dx*=*0,*(14)
*whereν*denotes the outer normal with respect*toG*n the left-hand side and with respect*toKR(0)*on the right-hand side of Eq. (14).
Now*letS*be a closed surface*in̂G andS*=*∂D*with a*domainD*⊂ R 3. In the case
*D*⊂*Ĝ*Gauß’s theorem yields ∫*S*
〈*αB, ν*
〉*d*=
∫*D*
div (*αB*
)*dx*=*0,*
in the other*case,G*⊂*D,*using Eq. (14) and Gauß’s theorem applied*toD \G*leads to ∫*S*
〈*αB, ν*
〉*d*=
∫*∂G*
〈*αB, ν*
〉*d*+
∫
*D\G*div
(*αB*
)*dx*=*0.*(15)
*Here,ν*denotes the outer normal with respect*toD atS*=*∂D*on the left-hand side of the first equation of (15) and with respect*toG*at*∂G*on the right-hand side.ut
**5. Convergence of****theα-Iteration**
**Lemma****5.1.LetG,***Ĝ, µ,M be as in Definition4.1, ρ0, T > 0,B*∈*C1,λ(Ĝ,R3) with〈 B, ν*
〉 ≥*ρ0. Let γ (., s) be the solution of the field line equation in Definition4.1 with respect toB and let*
*Dγ*:= (*∂γ*
*∂t*
∣∣∣∣*∂γ*
*∂s(1)*
∣∣∣∣*∂γ*
*∂s(2)*
)
*denote the Jacobian matrix ofγ . Then there exist(in both variables) monotonically increasing functionsκ1, κ2*: R
0+ ×R 0+ → R
0+*depending onG,ρ0,M, µ andλ but not onB, T , such that*
‖*Dγ ‖Cλ(0)*≤*κ1*( ‖*B*‖
*C1,λ(Ĝ) , T*
)*and*
‖*(Dγ )−1 ‖Cλ(0)*≤*κ2*( ‖*B*‖
*C1,λ(Ĝ) , T*
)*,*
*with 0*:=*0(M, B, T ).*
*Proof.*For simplicity we do not distinguish*betweenT andT (s) whereT (s) < T*(cf. Def. 4.1). Otherwise we would have to*replaceT*by*min(T (s), T ).*This simplification is possible*sinceB*can always be extended to the entire spaceR
3 with the consequence*T (s)*= ∞. (a) According to Lemma 4.3
*∂γ*
*∂s(1)*
(*. , s*
) and
*∂γ*
*∂s(2)*
(*. , s*
)
124 R. Kaiser, M. Neudert, W. von Wahl
are solutions of the linear equation
*ω̇(t)*=*DB*(*γ (t, s)*
) ·*ω(t),*(16)
*whereDB*is the Jacobian matrix*ofB.*Since
*∂*
*∂t*
(*∂γ*
*∂t*
) =*∂*
*∂t*
(*B(γ )*
) =*DB(γ )*·*∂γ ∂t*
*,*
*γ̇*=*∂γ ∂t*
is also a solution of Eq. (16). Furthermore, we have
*∣∣detDγ (0, s)*∣∣ =
∣∣∣∣ 〈*∂γ*
*∂t (0, s)*
*∂γ*
*∂s(1) (0, s)× ∂γ*
*∂s(2) (0, s)*
〉∣∣∣∣ ≥*ρ1*
〈*B*
(*µ(s)*
)*, ν̃(s)*
〉 ≥*ρ1ρ0 > 0.*
(17)
Thus {*∂γ*
*∂t (., s),*
*∂γ*
*∂s(1) (., s),*
*∂γ*
*∂s(2) (., s)*
}
is a fundamental system of solutions of Eq. (16)*in0(M, B).*(b) Applying Lemma 4.2*(withδ0*=*0,ϕ1*= 0) we get
∣∣∣∣*∂γ∂s(j) (t, s)*∣∣∣∣ ≤
∣∣∣∣*∂γ∂s(j) (0, s)*∣∣∣∣ · exp
(*‖DB‖ C0(Ĝ)*
·*t)*
≤ ‖*Dµ ‖C0(U)*· exp (*‖DB‖*
*C0(Ĝ)*·*T*)
for 0 ≤*t*≤*T*,*j*=*1,2.*Obviously there is*∣∣∣∣∂γ∂t (t, s)*
∣∣∣∣ ≤ ‖*B*‖*C0(Ĝ)*
*.*
(c) Now we*estimate[Dγ ]Cλ(0). Forω*=*∂γ ∂t ,*
*∂γ*
*∂s(1) ,*
*∂γ*
*∂s(2)*we have Eq. (16) and thus
for*s1, s2*∈*U*,
*ω̇(t,s2)−DB*(*γ (t, s1)*
) ·*ω(t, s2)*= [
*DB*(*γ (t, s2)*
) −*DB*(*γ (t, s1)*
) ] ·*ω(t, s2),*so*∣∣ω̇(t, s2)−DB*
(*γ (t, s1)*
) ·*ω(t, s2)*∣∣
≤ ‖*Dγ ‖C0(0)*· ‖*DB*‖*Cλ(Ĝ)*
· |*γ (t, s2)− γ (t, s1) |λ .*Lemma 4.2 yields then
|*ω(t, s2)− ω(t, s1)*| ≤ [ |*ω(0, s2)− ω(0, s1)*| + 2 ‖*Dγ ‖C0(0)*· ‖*DB*‖
*Cλ(Ĝ)*· sup*τ∈[0,T*[
|*γ (τ, s2)− γ (τ, s1) |λ ·T*] · exp
*(‖DB‖ C0(Ĝ)*
·*T*)*,*
On the Existence of Force-Free Magnetic Fields 125
where
|*γ (τ, s2)− γ (τ, s1)*| ≤ ‖*Dγ ‖C0(0)*·*|s2*−*s1|.*
In the*caseω*=*∂γ ∂t*
there is furthermore
|*ω(0, s2)*−*ω(0, s1)*| = |*B( µ(s2)*
) −*B*(*µ(s1)*
) | ≤ ‖*DB*‖
*C0(Ĝ)*· ‖*µ ‖Cλ(U)*·*|s2*−*s1|λ,*
otherwise*(ω*=*∂γ*
*∂s(j) , j*=*1,2)*
|*ω(0, s2)− ω(0, s1)*| ≤ ‖*Dµ ‖Cλ(U)*·*|s2*−*s1|λ.*So, in any case
|*ω(t, s2)− ω(t, s1)*|*|s2*−*s1|λ*≤ [ ‖*Dµ ‖Cλ(U)*+ ‖*DB*‖
*C0(Ĝ)*· ‖*µ ‖Cλ(U)*
+ 2 ‖*Dγ ‖1+λ C0(0)*
· ‖*DB*‖*Cλ(Ĝ)*
*·T*] · exp*(‖DB‖*
*C0(Ĝ)*·*T*)
for*s1*6=*s2,*0 ≤*t*≤*T*.*Sinceω*satisfies the linear differential equation (16), we have, moreover,
|*ω(t2, s)− ω(t1, s)*| |*t2*−*t1 |λ*≤*T 1−λ·*‖*ω̇ ‖C0(0)*
≤*T 1−λ·*‖*DB*‖*C0(Ĝ)*
· ‖*Dγ ‖C0(0)*
for 0 ≤*t1, t2*≤*T*,*t1*6=*t2, s*∈*U*. Obviously,*fort1, t2*∈*[0, T*],*t1*6=*t2, s1, s2*∈*U*,*s1*6=*s2*we have
|*ω(t2, s2)− ω(t1, s1)*|(*(t2*−*t1)2*+*|s2*−*s1|2*
*)λ/2*≤ |*ω(t2, s2)− ω(t2, s1)*|*|s2*−*s1|λ*+ |*ω(t2, s1)− ω(t1, s1)*|
|*t2*−*t1 |λ .*
Therefore
*[ω]Cλ(0)*≤*κ*(‖*B*‖
*C1,λ(Ĝ) , T*
)*,*
and applying the result of (b) we obtain
‖*Dγ ‖Cλ(0)≤ κ1*(‖*B*‖
*C1,λ(Ĝ) , T*
)*.*
(d) From the theory of linear ordinary differential equations it is well known that
*detDγ (t, s)*=*detDγ (0, s)*· exp {*t∫*
0
*traceDB*(*γ (t ′, s)*
)*dt*′
}*,*
and with the estimate (17) we have thus ∣∣*detDγ (t, s)*
∣∣ ≥*ρ1ρ0*· exp { − 3 ‖*DB*‖
*C0(Ĝ)*·*T*}
126 R. Kaiser, M. Neudert, W. von Wahl
for*(t, s)*∈*0(M, B, T ).*For the inverse*ofDγ*there holds the formula
(*Dγ*
)−1 = 1
*detDγ*· (*∂γ*
*∂s(1)*×*∂γ*
*∂s(2)*
∣∣∣∣*∂γ∂s(2)*×*∂γ*
*∂t*
*∣∣∣∣∂γ∂t*×*∂γ*
*∂s(1)*
)T
*.*(18)
Using the result of (b) we easily obtain the desired estimate*for‖(Dγ )−1‖C0(0).*
(e) Finally we look for an estimate*for[(Dγ )−1]Cλ(0).*Applying the estimates (6) and (8) we have
[(*Dγ*
)−1]*Cλ(0)*
≤ ‖ (*detDγ*
)−1*‖C0(0)*· [(*detDγ*
) · (*Dγ*
)−1]*λ*
+ + [(
*detDγ*)−1]
*λ*· ‖ (
*detDγ*) · (
*Dγ*)−1*‖C0(0)*
≤ ‖ (*detDγ*
)−1*‖C0(0)*·*[(detDγ*) · (
*Dγ*)−1]
*λ*
+ [*detDγ*
]*λ*· ‖ (
*detDγ*)−1 ‖2
*C0(0)*
· ‖ (*detDγ*
) · (*Dγ*
)−1*‖C0(0),*
and using Eq. (18), the estimate (7) and the results of (b),(c) and (d) we finally obtain
‖ (*Dγ*
)−1*‖Cλ(0)≤ κ2*(‖*B*‖
*C1,λ(Ĝ) , T*
)*.*ut
**Lemma****5.2.AssumeG,***Ĝ, µ,M as in Definition 4.1,T , ρ0 > 0, B*∈*C1,λ(Ĝ,R3), the configuration*
*(M, B*)
*admissible with parameters(ρ0, T , δ) andα0*∈*C1,λ*0*(M,R).*
*Letα be the trivial extension of the solution of the initial value problem(see Lemma 4.9)〈∇α,B〉*=*0, α ∣∣M*=*α0.*
*Then there exists a monotonically increasing functionκ3*: R + 0 × R
+ 0 → R
+ 0*depending*
*onG,µ,M, ρ0, λ, but not onB, T , α0, with the following property:*
‖*α*‖*Cλ(Ĝ)*
*,*‖*α*‖*C1,λ(Ĝ)*
≤ ‖*α0 ‖C1,λ(M) ·κ3*( ‖*B*‖
*C1,λ(Ĝ) , T*
)*,*
*where‖ α0 ‖C1,λ(M)*:= ‖*α0*
*Proof.*Using the notation of Theorem 4.8 we*setψhom*=*α, ψinh*= 0 and define
*ϑ(t, s)*:=*α*(*γ (t, s)*
) =*α0(µ(s)),*(19)
*whereγ*stands as in Theorem 4.8 for the solution of the field line equation*ofB.*We use again the*abbreviation0*:=*0(M, B, T ).*(a) We first*estimateϑ*. It is easy to see that
*∂ϑ*
*∂t*= 0 and ‖*ϑ ‖C0(0)*= ‖*α0 ‖C0(M),*
and from (19) we have
‖*∇ϑ ‖Cλ(0)≤*‖*α0 ‖C1,λ(M) .*
(b) Next we*estimateα.*Using the*abbreviation3*:=*3(M, B, T )*=*3(M, B)*we immediatly see
‖*α ‖C0(3)=*‖*ϑ ‖C0(0),*
On the Existence of Force-Free Magnetic Fields 127
and with
*∇ϑ*= (*Dγ*
)T · (*(∇α)*
we get
‖*∇α ‖C0(3)≤*‖ (*Dγ*
)−1*‖C0(0)*· ‖*∇ϑ ‖C0(0) .*
Now*consider[∇α]Cλ(3).*For*t1, t2*∈*[0, T*],*s1, s2*∈*U*with*(t1, s1)*6=*(t2, s2)*we obtain∣∣*(∇α)(*
*γ (t1, s1)*) −*(∇α)(*
*γ (t2, s2)*) ∣∣
|*γ (t1, s1)− γ (t2, s2) |λ*
≤ ‖ (*Dγ*
)−1*‖λ C0(0)*
∣∣ (*((Dγ )T)−1*·*∇ϑ)*
*(t1, s1)*− (*((Dγ )T)−1*·*∇ϑ)*
*(t2, s2)*∣∣
|*(t1*−*t2)2*+*|s1*−*s2|2 |λ/2 ,*
thus*[∇α] Cλ(3)*
≤ ‖ (*Dγ*
)−1*‖λ C0(0)*
· ‖ (*Dγ*
)−1*‖Cλ(0)*· ‖*∇ϑ ‖Cλ(0) .*Since*suppα*⊂*3*Lemma 5.1 and part (a) of the proof imply
‖*α*‖*C1,λ(Ĝ)*
= ‖*α ‖C1,λ(3)≤ ‖α0‖C1,λ(M)*·*κ(‖B‖ C1,λ(Ĝ)*
*, T*)
with*κ*being a function as described in the lemma. Furthermore, there holds the estimate (note*that3*=*3(M, B))*
‖*α*‖*Cλ(Ĝ)*
≤ ‖*α ‖Cλ(3)≤*≤ (
1 +*(diam3)1−λ)·*‖*α ‖C1,λ(3)*
≤ ( 1 +*(‖B‖*
*C0(Ĝ)*·*T*+*diamM)1−λ)*
·*‖α0‖C1,λ(M)*·*κ(‖B‖ C1,λ(Ĝ)*
*, T*)*.*
Here use has been made of Lemma 4.6. The last two estimates contain the statement of the lemma. ut**Lemma****5.3.AssumeG,***Ĝ, µ,M as in Definition4.1, ρ0, T , δI , δII > 0, BI , BII*∈*C1,λ(Ĝ,R3), α0*∈*C*
*1,λ*0*(M,R) and the configuration(M, Bj ) to be admissible with*
*parameters(ρ0, T , δj ), j*=*I, II, respectively. Letαj denote the trivial extension of the solution of the initial value problem〈∇αj , Bj*〉 = 0*in 3j*:=*3(M, Bj ), αj*
*∣∣M*=*α0, j*=*I, II .*
*Then there exist monotonically increasing functionsκ4, κ5*: R 0+×R
0+ → R 0+*depending*
*onG,µ,M, ρ0, but not onBj , α0, T , satisfying*
‖*αII*−*αI ‖Cλ(3II )*≤*‖α0‖C1,λ(M)*·*κ4*
*(‖BI‖C1,λ(Ĝ) , T*
) ·*κ4*
*(‖BII‖C1,λ(Ĝ) , T*
) ·*‖BI*−*BII‖Cλ(3II ) ,*
‖*αII*−*αI*‖*Cλ(Ĝ)*
≤*‖α0‖C1,λ(M)*·*κ5 (‖BI‖C1,λ(Ĝ)*
*, T*)
·*κ5 (‖BII‖C1,λ(Ĝ)*
*, T*) ·*‖BI*−*BII‖C1,λ(Ĝ)*
*.*
128 R. Kaiser, M. Neudert, W. von Wahl
*Proof. Let3j*:=*3(M, Bj ), 0j*:=*0(M, Bj ) andγj*denote the solution of the field line equation with respect*toBj*according to Definitions 4.1 and*4.5,j*=*I, II*. (a) We have
(*αII−αI*
)(*γII (t, s)*
) =*α0(µ(s))*−*αI*(*γII (t, s)*
)
= −*t∫*
0
*〈∇αI (γII (τ, s)*)*, γ̇II (τ, s)*
〉*dτ*
= −*t∫*
0
*〈∇αI (γII (τ, s)*)*,*(*BII*−*BI*
)(*γII (τ, s)*
) 〉*dτ*
and thus (cf. Lemma 5.2)
‖*αII*−*αI ‖C0(3II )*≤*T*
2 · ‖*∇αI*‖
*C0(Ĝ)*· ‖*BI*−*BII ‖C0(3II )*
≤ ‖*α0 ‖C1,λ(M) ·κ3(‖ BI*‖*C1,λ(Ĝ)*
*, T*· ‖*BII*−*BI ‖Cλ(3II ) .*
(b) In order to*estimate[αII*−*αI ]λ consider∣∣(αII*−*αI*)(*γII (t1, s1)*
) − (*αII*−*αI*
)(*γII (t2, s2)*
*)∣∣∣∣γII (t1, s1)− γII (t2, s2) ∣∣λ*
≤ ‖*(DγII )*−1*‖λ*
*C0(0II )*
·*{∣∣(αII*−*αI*
)(*γII (t1, s1)*
) − (*αII*−*αI*
)(*γII (t1, s2)*
)∣∣*|s1*−*s2|λ*
+*∣∣(αII*−*αI*
)(*γII (t1, s2)*
) − (*αII*−*αI*
)(*γII (t2, s2)*
)∣∣ |*t1*−*t2 |λ*
}*.*
(20)
Next we estimate the first term in curly brackets on the right-hand side of Eq. (20),
1
*|s1*−*s2|λ*·*∣∣(αII*−*αI*)(*γII (t1, s1)*
) − (*αII*−*αI*
)(*γII (t1, s2)*
)∣∣
= 1
*|s1*−*s2|λ*· ∣∣∣∣*t1∫*
0
*{〈∇αI , (BI*−*BII )*〉(*γII (τ, s1)*
)
−*〈∇αI , (BI*−*BII )*〉(*γII (τ, s2)*
)}*dτ*
∣∣∣∣ ≤ ‖*DγII ‖λ*
*C0(0II )*·*t1∫*
0
*1∣∣γII (τ, s1)− γII (τ, s2) ∣∣λ*
·*∣∣〈∇αI , (BI*−*BII )*〉∣∣∣∣*γII (τ,s1)*
*γII (τ,s2)*
*∣∣dτ*≤ ‖*DγII ‖λ*
*C0(0II ) ·T*· ‖*∇αI*‖
*Cλ(Ĝ)*· ‖*BII*−*BI ‖Cλ(3II )*
≤ ‖*α0 ‖C1,λ(M) ·κ3*(‖*BI*‖
*C1,λ(Ĝ) , T*
) ·*κλ1*
(‖*BII*‖*C1,λ(Ĝ)*
*, T*) ·*T*· ‖*BII*−*BI ‖Cλ(3II )*
*.*
On the Existence of Force-Free Magnetic Fields 129
For the last inequality we have used Lemmas 5.1 and 5.2. Finally we consider the second term in the curly brackets on the right-hand side of the estimate (20),
1
|*t1*−*t2 |λ*·*∣∣(αII*−*αI*)(*γII (t1, s2)*
) − (*αII*−*αI*
)(*γII (t2, s2)*
)∣∣ = 1
|*t1*−*t2 |λ*·*∣∣αI (γII (t1, s2)*) −*αI*
(*γII (t2, s2)*
)∣∣
= 1
|*t1*−*t2 |λ*· ∣∣*t2∫ t1*
*〈∇αI , (BII*−*BI )*〉(*γII (τ, s2)*
)*dτ*
∣∣
≤ |*t1*−*t2 |1−λ*· ‖*∇αI*‖*C0(Ĝ)*
· ‖*BII*−*BI ‖C0(3II )*
≤*T 1−λ·*‖*α0 ‖C1,λ(M) ·κ3*(‖*BI*‖
*C1,λ(Ĝ) , T*
)· ‖*BII*−*BI ‖C0(3II ) .*
Again, we have used Lemma 5.2 for the last inequality. From both estimates we conclude
*[αII*−*αI ]Cλ(3II )*≤ ‖*α0 ‖C1,λ(M)*·*κ3*
(‖*BI*‖*C1,λ(Ĝ)*
*, T*)
·*κ(‖ BII*‖*C1,λ(Ĝ)*
*, T*) · ‖*BII*−*BI ‖Cλ(3II )*
*,*
with*κ*being a monotonically increasing function. The first estimate in Lemma 5.3 follows now from (a) and (b). (c) Now**letC**be the convex hull*of3II*. Applying the mean value theorem we obtain
*[B]Cλ(3II )*≤ √
3 ·*(diam3II ) 1−λ·*‖*B***‖C1(C)***.*
The factor √
3 is due to the application of the mean value theorem to each of the three components*ofB.*Clearly
‖*B***‖C1(C)≤‖***B ‖C1+λ(R3)≤ c1*‖*B*‖*C1,λ(Ĝ)*
*.*
So we have the estimate*(here3*:=*3II*)
‖*B ‖Cλ(3)*≤ √*3c1*
( 1 +*(diam3)1−λ)·*‖*B*‖
*C1,λ(Ĝ) ,*(21)
and with Lemma 4.6*(3II*:=*3(M, BII ))*
‖*BI−BII ‖Cλ(3II )*≤ √
3*c1*( 1 +*(diam3II )*
*1−λ)·*‖*BI*−*BII*‖*C1,λ(Ĝ)*
≤ √ 3*c1*
( 1 +*(‖BII‖C0(Ĝ)*
·*T*+*diamM)1−λ)*· ‖*BI*−*BII*‖*C1,λ(Ĝ)*
*.*
Thus the first part of Lemma 5.3 yields
‖*αI*−*αII ‖Cλ(3II )*≤ ‖*α0 ‖C1,λ(M) ·κ4*
*(‖BI‖C1,λ(Ĝ) , T*
) ·*κ(‖BII‖C1,λ(Ĝ)*
*, T*)· ‖*BI*−*BII*‖
*C1,λ(Ĝ)*
with*κ*being again a monotonically increasing function. Since
*supp(αI*−*αII )*⊂*3(M, BI )*∪*3(M, BII )*
and using the corresponding estimate with*labelsI andII*interchanged we finally obtain the second estimate in Lemma 5.3.ut
130 R. Kaiser, M. Neudert, W. von Wahl
We are now in the position to present the main result of this article which is the proof of the convergence of the iteration scheme and by this the construction of a force-free magnetic field in the exterior domain.
**Theorem****5.4.LetG,***Ĝ,M, µ be as in Definition4.1, ρ0, T , δ > 0,*0 ∈*G, B0*∈*C1,λ(Ĝ,R3) with*
*divB0*= 0*and curlB0*=*0, |B0(x)|*=*O(|x|−2), |x|*→*∞.*
*Suppose the configuration (M, B0*
)*to be admissible with parameters(ρ0, T , δ). Then*
*there existsη > 0depending onG,M, µ, ρ0, T , δ and‖ B0*‖*C1,λ(Ĝ)*
*, with the following property: If α0*∈*C1,λ*
0*(M,R) with ‖α0‖C1,λ(M) < η, then the iteration scheme*
*(i) αn solves 〈∇αn, Bn−1*
〉 =*0, αn ∣∣M*=*α0 according to Th.4.8,*
*(ii ) jn*:=*αnBn−1,*
*(iii ) Bn solvescurlBn*=*jn,divBn*=*0,*〈*Bn, ν*
〉∣∣*∂G*
= 〈*B0, ν*
〉∣∣*∂G*
*according to Theorem3.1,*
*converges in the following sense: There existB*∈*C1,λ(Ĝ,R3) andα*∈*Cλ(Ĝ,R) with*
‖*Bn*−*B*‖*C1,λ(Ĝ)*
→*0, n*→*∞,*
‖*αn*−*α*‖*Cλ(Ĝ)*
→*0, n*→*∞,*
*divB*=*0, curlB*=*αB in Ĝ,〈 B, ν*
〉∣∣*∂G*
= 〈*B0, ν*
〉∣∣*∂G ,*
*|B(x)|*=*O (|x|−2), |x|*→*∞,*
*α ∣∣M*=*α0.*
*Proof.*We use the*abbreviation3n*:=*3(M, Bn)*for*n*∈ N0.*Let%*∈*]1,3[*,*c0*be the constant depending*onλ, %,G*in Theorem*3.2,c1*the constant for the extensions*ofB0, Bn*to R
3 described at the beginning of Sect. 3,
*K1*:= 2 ‖*B0*‖*C1,λ(Ĝ)*
*, R0*:=*diamG,*
*andη >*0 so small that
*c0c1*√
3 · (*(R0 +K1T*+*diamM)%*+ 1
) · ( 1 +*(K1T*+*diamM)1−λ)*
·*ηK1*·*κ3*(*K1, T*
)*<*min
(1*4K1,*
*δ 4c1T*
*e*− 1
*2c1K1T*) (22)
and
*c0c1*√
3 · (*(R0 +K1T*+*diamM)%*+ 1
) · (
1 +*(2K1T*+*2diamM)1−λ)*· [*K1*·*κ2*
5
(*K1, T*
) +*κ3*(*K1, T*
)] ·*η <*1
2*.*
(23)
On the Existence of Force-Free Magnetic Fields 131
Now*supposeα0*∈*C1,λ*0*(M) satisfying‖α0‖C1,λ(M) < η. α1*is the trivial extension of
the unique solution of the initial value problem
〈*α1, B0*
〉 = 0*in30, α1 ∣∣M*=*α0*
according to Theorem*4.8.B1 −B0*is then the unique solution of the Neumann problem
curl*(B1*−*B0)*=*α1B0,*div*(B1*−*B0)*=*0,*〈*B1*−*B0, ν*
〉∣∣*∂G*
= 0 (24)
according to Theorem 3.1. Note*thatα1*has compact support and thus the asymptotic condition*onw*in Theorem 3.1 is satisfied.
*Forx*∈*3n*we have, while 0∈*G,*
|*x*|≤*diamG+ diam3n*
and therefore for a bounded*functionq,*
*|x|%|q(x)|*≤ (*diamG+ diam3n*
*)%·*‖*q*‖∞*.*(25)
Applying Lemma 4.6 we obtain
|‖*α1B0 |‖%≤ (R0 +K1T*+*diamM)%·*‖*α1B0*‖*C0(Ĝ)*
*.*
Because of Theorem 3.2 there holds
‖*B1−B0*‖*C1,λ(Ĝ)*
≤*c0*· ( 1 +*(R0 +K1T*+*diamM)%*
)· ‖*α1B0*‖*Cλ(Ĝ)*
≤*c0c1*√
3 ( 1 +*(R0 +K1T*+*diamM)%*
)( 1 +*(K1T*+*diamM)1−λ)*
·*κ3 (‖B0‖C1,λ(Ĝ)*
*, T*)· ‖*α0 ‖C1,λ(M)*· ‖*B0*‖
*C1,λ(Ĝ) .*
For the last inequality we have applied Lemma 5.2 and the estimate (21). According to assumption (22) we have
‖*B1*−*B0*‖*C1,λ(Ĝ)*
*<*1
2 ‖*B0*‖
*C1,λ(Ĝ)*(26)
and
‖*B1*−*B0*‖*C1,λ(Ĝ)*
*< δ*
*4c1T*exp
(−1
2*c1K1T*
)*.*(27)
Taking into account the inequality (27) and Lemma 4.7 we realize that the*configuration(M, B1*)
is admissible with*parameters(ρ0, T ,*3*4δ).*
Next we show by induction*forn*∈ N:
‖*Bn*−*Bn−1*‖*C1,λ(Ĝ)*
*<*1
*2n−1*· ‖*B1*−*B0*‖*C1,λ(Ĝ)*
*<*1
*2n*· ‖*B0*‖
*C1,λ(Ĝ) ,*(28)
‖*Bn*−*B0*‖*C1,λ(Ĝ)*
*<*‖*B0*‖*C1,λ(Ĝ)*
·*n∑ j=1*
1
*2j ,*(29)
‖*Bn*−*B0*‖*C1,λ(Ĝ)*
*<*1
2 ·*δ*
*c1T*· exp
(−1*2c1K1T*
) ·*n∑ j=1*
1
*2j .*(30)
132 R. Kaiser, M. Neudert, W. von Wahl
The*casen*= 1 corresponds to the inequalities (26) und (27) which have already been*proved.Bn+1*−*Bn*is for*n*∈ N the solution of
curl*(Bn+1*−*Bn)*=*αn+1Bn*−*αnBn−1,*div*(Bn+1*−*Bn)*=*0,〈 Bn+1*−*Bn, ν*
〉∣∣*∂G*
= 0
according to Theorem 3.1,*whereαn, αn+1*are the unique solutions*of〈∇αn, Bn−1*〉 = 0*in3n−1, αn*
*∣∣M*=*α0,〈∇αn+1, Bn*〉 = 0*in3n, αn+1*
*∣∣M*=*α0*
according to Theorem 4.8*withαn*= 0 in*Ĝ \3n−1 andαn+1*= 0 in*Ĝ \3n.*Now suppose the inequalities (28), (29) and (30) to be proved for indices*1, . . . , n.*
Therefore
‖*Bn*‖*C1,λ(Ĝ)*
*<*‖*B0*‖*C1,λ(Ĝ)*
·*n∑ j=0*
1
*2j < K1,*
‖*Bn−1*‖*C1,λ(Ĝ)*
*<*‖*B0*‖*C1,λ(Ĝ)*
·*n−1∑ j=0*
1
*2j < K1,*
(31)
‖*Bn*−*B0*‖*C1,λ(Ĝ)*
*<*1
2 ·*δ*
*c1T*· exp
(−1
2*c1K1T*
) ·*n∑ j=1*
1
*2j*
≤ (1
2 ·*n∑ j=1*
1
*2j*
) ·*δ*
*c1T*· exp
( −1
2*c1*‖*B0*‖
*C1(Ĝ) ·T*
)*,*
thus (cf. Lemma 4.7) the configuration*(M, Bn*
) is admissible with parameters(
*ρ0, T , (1*− 1 2
*∑n j=1 2−j )δ*
) and
‖*Bn−1*−*B0*‖*C1,λ(Ĝ)*
*<*1
2 ·*δ*
*c1T*· exp
(−1
2*c1K1T*
) ·*n−1∑ j=1*
1
*2j*
*<*(1
2 ·*n−1∑ j=1*
1
*2j*
) ·*δ*
*c1T*· exp
( −1
2*c1*‖*B0*‖
*C1(Ĝ)*·*T*
)*,*
thus*(M, Bn−1*
) is admissible with parameters
(*ρ0, T , (1−*1
2
*∑n−1 j=1 2−j )δ*
) . In particular,
the configurations*(M, Bn*
) and
*(M, Bn−1*)
are admissible with parameters (*ρ0, T ,*
*δ*2
) .
Using Theorem 3.2 we have the estimate
‖*Bn+1*−*Bn*‖*C1,λ(Ĝ)*
≤*c0*·*(‖αn+1Bn*−*αnBn−1‖Cλ(Ĝ)+*|‖*αn+1Bn*−*αnBn−1 |‖%*)*.*
Since*supp(αn+1Bn−αnBn−1 )*⊂*3n∪3n−1*this means in accordance with inequality (25) and Lemma 4.6
‖*Bn+1*−*Bn*‖*C1,λ(Ĝ)*
≤*c0*· ( 1 +*(R0 +K1T*+*diamM)%*
)· ‖*αn+1Bn*−*αnBn−1*‖*Cλ(Ĝ)*
*.*(32)
On the Existence of Force-Free Magnetic Fields 133
Furthermore (cf. (7)),
‖*αn+1Bn*−*αnBn−1*‖*Cλ(Ĝ)*
= ‖*αn+1Bn*−*αnBn−1 ‖Cλ(3n∪3n−1)*
≤ ‖*αn+1*−*αn*‖*Cλ(Ĝ)*
· ‖*Bn ‖Cλ(3n∪3n−1)*
+ ‖*αn ‖Cλ(3n−1)*· ‖*Bn*−*Bn−1 ‖Cλ(3n∪3n−1)*
*,*
(33)
and with Lemma 5.3 in inequalities (31),
‖*αn+1*−*αn*‖*Cλ(Ĝ)*
≤ ‖*α0 ‖C1,λ(M) ·κ2*5
(*K1, T*
)· ‖*Bn*−*Bn−1*‖*C1,λ(Ĝ)*
*.*(34)
*Obviously3n ∪3n−1*is pathwise connected. Since
diam (*3n ∪3n−1*
) ≤*2K1T*+ 2*diamM*we have according to inequality (21),
‖*Bn ‖Cλ(3n∪3n−1)*≤ √
*3c1*( 1 +*(2K1T*+*2diamM)1−λ)·*‖*Bn*‖
*C1,λ(Ĝ)*
≤ √*3c1*
( 1 +*(2K1T*+*2diamM)1−λ) ·K1,*
(35)
and correspondingly
‖*Bn−Bn−1 ‖Cλ(3n∪3n−1)*
≤ √ 3*c1*· (
1 +*(2K1T*+ 2*diamM)1−λ)·*‖*Bn*−*Bn−1*‖*C1,λ(Ĝ)*
*.*(36)
The inequalities (34), (35), (36) inserted into (33) together with Lemma 5.2 furnish
*‖αn+1Bn*−*αnBn−1‖Cλ(Ĝ)*≤ √
*3c1*· ( 1 +*(2K1T*+*2diamM)1−λ)*
· [*K1κ*
2*5(K1, T )+ κ3(K1, T )*
] ·*‖α0‖C1,λ(M)*·*‖Bn*−*Bn−1‖C1,λ(Ĝ) .*
Using inequalities (32), (23)*and‖α0‖C1,λ(M) < η*we obtain
‖*Bn+1*−*Bn*‖*C1,λ(Ĝ)*
*<*1
2 · ‖*Bn*−*Bn−1*‖
*C1,λ(Ĝ) ,*(37)
and therefore
‖*Bn+1*−*Bn*‖*C1,λ(Ĝ)*
*<*1
*2n+1*· ‖*B0*‖*C1,λ(Ĝ)*
*.*
Thus
‖*Bn+1*−*B0*‖*C1,λ(Ĝ)*
*<*‖*B0*‖*C1,λ(Ĝ)*
·*n+1∑ j=1*
1
*2j ,*
and in particular
‖*Bn+1*‖*C1,λ(Ĝ)*
*<*2 ‖*B0*‖*C1,λ(Ĝ)*
=*K1.*
134 R. Kaiser, M. Neudert, W. von Wahl
According to inequalities (37), (28) and (27) we have
‖*Bn+1*−*Bn*‖*C1,λ(Ĝ)*
*<*1
*2n*· ‖*B1*−*B0*‖
*C1,λ(Ĝ)*
*<*1
2 ·*δ*
*c1T*· exp
(−1
2*c1K1T*
) · 1
2 · 1
*2n ,*
(38)
and therefore using inequalities (30) and (38)
‖*Bn+1*−*B0*‖*C1,λ(Ĝ)*
≤ ‖*Bn+1*−*Bn*‖*C1,λ(Ĝ)*
+ ‖*Bn*−*B0*‖*C1,λ(Ĝ)*
≤ 1
2 ·*δ*
*c1T*· exp
(−1
2*c1K1T*
) · (1
2 · 1
*2n*+
*n∑ j=1*
1
*2j*)
= 1
2 ·*δ*
*c1T*· exp
(−1
2*c1K1T*
) ·*(n+1∑ j=1*
1
*2j*)*.*
So*(M, Bn+1*
) is admissible with parameters
(*ρ0, T , (1*− 1
2
*∑n+1 j=1 2−j )δ*
) (cf. Lemma
4.7), and also with parameters (*ρ0, T ,*
*δ*2
) . Now the estimates (28), (29) and (30) are valid
for all*n*∈ N. Due to inequality (28)*form, n*∈ N there holds the estimate
‖*Bn+m*−*Bn*‖*C1,λ(Ĝ)*
≤*m∑ k=1*
‖*Bn+k*−*Bn+k−1*‖*C1,λ(Ĝ)*
≤*m∑ k=1*
1
*2n+k*· ‖*B0*‖
*C1,λ(Ĝ)*≤ 1
*2n*· ‖*B0*‖
*C1,λ(Ĝ) .*
So (*Bn*
)*n∈N*
is a Cauchy sequence in the Banach*spaceC1,λ(Ĝ,R3)*and converges to
*someB*∈*C1,λ(Ĝ,R3)*with respect to the norm‖*.*‖*C1,λ(Ĝ)*
. Because of
‖*αn+1*−*αn*‖*Cλ(Ĝ)*
≤*η*·*κ2*5
(*K1, T*
)· ‖*Bn*−*Bn−1*‖*C1,λ(Ĝ)*
(see Lemma 5.3) (*αn*
)*n∈N*
is a Cauchy sequence*inCλ(Ĝ,R).*We set
*α*:= lim*n→∞αn,*
and obtain
‖*curlB*−*αB*‖*Cλ(Ĝ)*
= lim*n→∞*‖*curlBn*−*αnBn−1*‖
*Cλ(Ĝ)*=*0,*
with*divB*= 0.The configuration*(M, B*
) is thus admissible with parameters
(*ρ0, T ,*
*δ*2
) .
Finally, there exists*someR >*0 such that
*G*∪*suppαB*⊂*KR(0).*
In the exterior*ofKR(0)*the*fieldB*is harmonic satisfying
*|B(x)|*=*O (|x|1−%), |x|*→*∞.*
From Theorem 3.1 we know then
*|B(x)|*=*O (|x|−2), |x|*→*∞.*ut
On the Existence of Force-Free Magnetic Fields 135
*Remark 5.5.Suppose*in addition to the assumptions in Theorem 5.4∫*∂G*
〈*B0, ν*
〉*d*=*0,*
then the*fieldB*constructed there has the asymptotic behaviour
*|B(x)|*=*O (|x|−3), |x|*→*∞.*
*Proof.*We*chooseR >*0 sufficiently large, such that
*suppα*⊂*KR(0).*
*SinceB*is divergence-free*in̂G,*Gauß’s theorem yields∫*∂KR(0)*
〈*B(ξ),*
*ξ R*
〉*d*=
∫*∂G*
〈*B(ξ), ν(ξ)*
〉*d*=
∫*∂G*
〈*B0(ξ), ν(ξ)*
〉*d*=*0.*
Because*ofB*being harmonic inR3*\KR(0),*we deduce from Theorem 3.1,
*|B(x)|*=*O (|x|−3), |x|*→*∞.*ut
*Remarks 5.6.(a)*The interior problem can be treated analogously, if the additional condition
∫*∂G*
*〈B0, ν〉d*= 0 is supposed.
(b) To extend the result of Theorem 5.4 to the case of a multiply connected*domainĜwith*first*Betti-numberñ*(see Remark 3.3(b)) step (iii) in the iteration has to be completed by the condition∫
*∂G*
*〈ν*×*Bn,***zj***〉d*= ∫*∂G*
*〈ν*×*B0,***zj***〉d*=:*0j , j*=*1, . . . , ñ,*
i.e. the circulations have to be fixed during the iteration. To obtain an asymptotic decay of*|B(x)|*=*O(|x|−3), |x|*→ ∞, all*0j*have to vanish.
*Acknowledgements.The*authors would like to thank B. J. Schmitt and J. J. Aly for discussions and valuable comments, respectively, on the material presented in this paper.
The authors were supported by the DFG-researchers group: Equations of Hydrodynamics.
**References**
[Al1] Aly, J.J.: On some properties of force-free magnetic fields in infinite regions of space. Astrophys.J.**283,**349–362 (1984)
[Al2] Aly, J.J.: Eruptive processes in the solar corona. In: Lynden-Bell, D.*(ed.),Cosmical Magnetism.*Cambridge: Publ. Inst. of Astronomy, 1993, pp.7–15
[Bin] Bineau, M.: On the existence of Force-Free Magnetic Fields. Comm. Pure Appl.**Math.25,**77–84 (1972)
[Bra] Bray, R.J. et.al.: Plasma loops in the solar corona. Cambridge: Cambridge University Press, 1991 [Ch1] Chandrasekhar, S.: On force-free magnetic fields. Proc. Nat. Acad.**Sci.42,**1–5 (1956) [Ch2] Chandrasekhar,*S.:Hydrodynamic and hydromagnetic stability.*Oxford: Clarendon, 1961 [CC] Chang, H.M., Carovillano, R.L.: Non Linear Force Free Magnetic Fields with Chosen Symmetry.
Bull. AAS**13,**909 (1981)
136 R. Kaiser, M. Neudert, W. von Wahl
[GT] Gilbarg, D., Trudinger,*N.S.:Elliptic Partial Differential Equations of Second Order.*2.Aufl. Berlin– Heidelberg–New York–Tokyo: Springer-Verlag, 1983
[GR] Grad, H., Rubin, H.: Hydromagnetic Equilibria and Force-Free Fields.*In:Proc. 2nd Intern. Conf. Peaceful Uses of Atomic Energy,*Vol.**31,**Geneva: United Nations, 1958, pp. 190–197
[Gra] Grad, H.: Theory and applications of the nonexistence of simple toroidal plasma equilibrium. In: Cercignani, C., Rionero, S., Tessarotto, M.*(eds.),Proceedings of the Workshop on Mathematical Aspects of Fluid and Plasma Dynamics,*Trieste, Università Degli Studi Di Trieste, Facultà Di Scienze, Instituto Di Mechanica, Trieste, 1984, pp. 253–282
[Kr1] Kress, R.: Ein Neumannsches Randwertproblem bei kraftfreien Feldern. Meth. Verf. math.**Phys.7,**81–97 (1972)
[Kr2] Kress, R.: A remark on a boundary value problem for force free fields. Z. Angew. Math.**Phys.28,**715–722 (1977)
[Lor] Lortz, D.: Über die Existenz toroidaler magnetohydrostatischer Gleichgewichte ohne Rotationstrans-formation. Z. Angew. Math.**Phys.21,**196–211 (1970)
[Low] Low, B.C.: Magnetic field configurations associated with polarity intrusion in a solar active region. Solar**Phys.115,**269–276 (1988)
[LS] Lüst, R., Schlüter, A.: Kraftfreie Magnetfelder. Z. f.**Astrophys.34,**263–282 (1954) [NvW] Neudert, M., von Wahl, W.: Asymptotic behaviour of the div-curl-problem in exterior domains. To
be published (2000) [Pic] Picard, R.: Ein Randwertproblem in der Theorie kraftfreier Magnetfelder. Z. Angew. Math.**Phys.27,**
169–180 (1976) [Pri] Priest,*E.R.:Solar Magnetohydrodynamics.*Dordrecht: Reidel, 1982 [Sak] Saks, R.S.: On the boundary value problem for the system*rotu*+*λu*=*h.*Differ.**Eqs.8,**97–102
(1972) [Spi] Spitzer, L.jr.: The Stellarator concept. Phys.**Fluids1,**253 (1958) [Tay] Taylor, J.B.: Relaxation of toroidal plasma and generation of reversed magnetic fields. Phys. Rev.
Lett.**33,**1139–1141 (1974) [YG] Yoshida, Z., Giga, Y.: Remarks on spectra of operator rot. Math.**Z.204,**235–245 (1990) [Wol] Woltjer, L.: A theorem on force-free magnetic fields. Proc. Nat. Acad.**Sci.44,**489–491 (1958)
Communicated by A. Kupiainen