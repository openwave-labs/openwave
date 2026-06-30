
### Local Representation and Construction
### of Beltrami Fields
N. Sato1 and M. Yamada1
1Research Institute for Mathematical Sciences, Kyoto University, Kyoto 606-8502, Japan
Email: sato@kurims.kyoto-u.ac.jp
September 11, 2018
Abstract
A Beltrami field is an eigenvector of the curl operator. Beltrami fields describe steady flows in fluid dynamics and force free magnetic fields in plasma turbulence. By application of the Lie-Darboux theorem of differential geoemtry, we prove a local representation theorem for Beltrami fields. We find that, locally, a Beltrami field has a standard form amenable to an Arnold-Beltrami-Childress flow with two of the parameters set to zero. Furthermore, a Beltrami flow admits two local invariants, a coordinate representing the physical plane of the flow, and an angular momentum-like quantity in the direction across the plane. As a consequence of the theorem, we derive a method to construct Beltrami fields with given proportionality factor. This method, based on the solution of the eikonal equation, guarantees the existence of Beltrami fields for any orthogonal coordinate system such that at least two scale factors are equal. We construct several solenoidal and nonsolenoidal Beltrami fields with both homogeneous and inhomogeneous proportionality factors.
1 Introduction
Let w ∈ C∞ (Ω) be a smooth vector field in a bounded domain Ω ⊂ R3. Object of the present study is the following system of first order partial differential equations:
w × (∇×w) = 0 in Ω. (1)
1
A solution w to (1) is called a Beltrami field. Evidently, a Beltrami field w 6= 0 in Ω is an eigenvector of the curl operator [1], i.e. it satisfies:
∇×w = αw in Ω, (2)
where α ∈ C∞ (Ω) is the proportionality factor (eigenvalue). Beltrami fields arise as stationary solutions of the Euler equations in fluid
dynamics [2, 3, 4] and as force free magnetic fields in magnetohydrodynamics [5, 6, 7]. Indeed, the steady ideal Euler equations at constant density,
(w · ∇)w = −∇P, ∇ ·w = 0 in Ω, (3)
reduce to the equation for a solenoidal Beltrami field,
w × (∇×w) = 0, ∇ ·w = 0 in Ω, (4)
whenever the pressure P is given by P = −w2/2. (4) is also the equation satisfied by a force free magnetic field. Beltrami fields occur in topologically constrained systems as well, where they are operators acting on a Hamilto-nian function to generate particle dynamics1[8].
In addition to physical applications, Beltrami fields play a key role in understanding topological properties of steady Euler flows [9]. In Refs. [10, 11, 12] it is shown that vortex tubes of arbitrarily complex topology can be realized by means of Beltrami fields. Despite the centrality of Beltrami fields in the study of fluid flows and plasma turbulence [13, 14], there are some aspects pertaining to their geometrical properties that have not yet been clarified. This can be understood by comparison with complex lamellar vector fields, i.e. vector fields w ∈ C∞ (Ω) with vanishing helicity density:
h = w · ∇ ×w = 0 in Ω. (5)
This condition is opposite to equation (1), in the sense that while (1) requires alignment between vector field and curl, equation (5) requires orthogonality. It is known (Frobenius theorem, see Ref. [15]) that a vector field obeying (5) is integrable. More precisely, for any x ∈ Ω, there exists a neighborhood U ⊂ Ω of x such that
w = λ∇C in U, (6)
where (λ,C) ∈ C∞ (U) are smooth functions in U . However, no local representation theorem is known for Beltrami fields. Furthermore, exception
1If w is a Beltrami operator, particle dynamics obeys the equation of motion ẋ = w ×∇H, with x the particle position in R3 and H a scalar function (the Hamiltonian).
2
made for standard Arnlod-Beltrami-Childress (ABC) flows [13, 16],
w = (A sin z + C cos y)∇x+ (B sinx+A cos z)∇y + (C sin y +B cosx)∇z, A,B,C ∈ R,
(7)
very few explicit examples of Beltrami fields are known, most of them with constant proportionality factors, i.e. such that ∇ × w = αw with α 6= 0 a real constant. This problem is both intrinsic and technical: on one hand the solution of (1) for general inhomogeneous proportionality factors is locally obstructed by the inhomogeneity, leading to Beltrami fields with lowregularity [17, 18, 19]. On the other hand any Beltrami field with nonzero proportionality factor is inevitably nonintegrable (it cannot satisfy h = 0). This automatically complicates the topology of the vector field, hiding the ‘natural’ form of the solution to (1). In terms of the Clebsch parametrization of a vector field [20, 21], this means that the representation of a Beltrami field with nonzero proportionality factor will always need more than two Clebsch parameters. Hence, Beltrami fields with nonzero proportionality factors are fully three-dimensional objects, and can be related to the notion of Reeb vector field in contact topology [22].
This paper is organized as follows. First, we prove the following local representation theorem for Beltrami fields by applying the Lie-Darboux theorem of differential geometry [23, 24, 25]:
Theorem 1. Let w ∈ C∞ (Ω) be a smooth vector field in a bounded domain Ω ⊂ R3 with h = w · ∇ ×w 6= 0 in Ω. Then w satisfies equation (1) if and only if for every x ∈ Ω there exists a neighborhood U ⊂ Ω of x and a local coordinate system (`, ψ, θ) ∈ C∞ (U) such that
cos θ sin θ ( |∇ψ|2 − |∇`|2
) = (∇` · ∇ψ)
( cos2 θ − sin2 θ
) , (8a)
sin θ ∇` · ∇θ + cos θ ∇ψ · ∇θ = 0, (8b)
and w = cos θ ∇ψ + sin θ ∇` in U. (9)
A direct consequence of Theorem 1 is that the flow generated by a Bel-trami field admits two local invariants, one representing the physical plane of the flow, the other conservation of an angular momentum-like quantity in the direction across the plane. The existence of two local invariants suggests a description of Beltrami flows in terms of Nambu brackets [26]. We also find that, if the Beltrami field is solenoidal, the proportionality factor is a function of the local invariants, giving an explanation of the helical flow paradox
3
[27]. In addition, Theorem 1 naturally leads to the following method for the construction of Beltrami fields with given proportionality factor α, i.e. such that ∇×w = αw.
Corollary 1. Let (`, ψ, θ) ∈ C1 (Ω) be an orthogonal coordinate system in a bounded domain Ω ∈ R3 such that, in Ω,
|∇θ| = |α|, (10a)
|∇`| = |∇ψ| , (10b)
where α ∈ C (Ω). Then, the vector fields
w = cos θ ∇ψ + sin θ ∇`, (11a)
w∗ = sin θ ∇ψ + cos θ ∇`, (11b)
are Beltrami fields in Ω with proportionality factors σ|α| and −σ|α| respectively, i.e. ∇ × w = σ|α|w and ∇ × w∗ = −σ|α|w∗, where σ = h/ |h| is the sign of the helicity density h = w · ∇ ×w.
In the second part of the paper we apply corollary 1 to construct Beltrami fields with homogeneous and inhomogeneous proportionality factors. Both solenoidal and non-solenoidal examples will be given.
2 Preliminaries
Consider equation (2). If α is a nonzero real constant, w is called a strong (or linear) Beltrami field. The helicity density h of the vector field w can be evalueated as
h = w · ∇ ×w = α w2. (12)
Here w = |w|. We shall say that a Beltrami field is nontrivial whenever h 6= 0 in Ω. If h 6= 0, we have w 6= 0 as well as ∇×w 6= 0 in Ω. Hence, the proportionality factor α can be expressed in terms of the helicity density h as
α = w−2 h = ĥ. (13)
Here ĥ = w−2 h = ŵ ·∇×ŵ represents the helicity density of the normalized vector field ŵ = w/w. In the following we will always be concerned with nontrivial Beltrami fields and adopt the notation
∇×w = ĥ w. (14)
4
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQGjha70BSWxEOI83KF-mKolcU7of4Flfa6MvFHtdl48HepyKMO6VkSqMX1SjhdECo1v69fmJxcnWu2906VTqIFJB3epS117_vHmwlkKKAKKrsY_hRQ8f6cvOvtnNoH6dml2FXXiNA=w1280-h756-v0?authuser=0)

If w is solenoidal, i.e. ∇·w = 0, the helicity density ĥ is an integral invariant of the flow generated by w,
∇ĥ ·w = 0. (15)
This can be verified by taking the divergence of equation (14). The simplest example of nontrivial Beltrami field in R3 is the vector field
w = sin z ∇x+ cos z ∇y, (16)
which satisfies w = 1, ∇×w = w, ĥ = 1, and ∇ ·w = 0. Furthermore, (16) can be written as
w = ∇z ×∇ (x cos z − y sin z) . (17)
Hence the flow generated by (16) has two integral invariants: the variable z which labels the planes where the flow lies, and the z-component of the angular momentum L = x × w, Lz = x cos z − y sin z. Figure 1 shows the behavior of the Beltrami field (16) on the level sets z = constant and Lz = constant. While the Beltrami field (16) is very simple, it encloses all local properties of general Beltrami fields. This is the content of Theorem 1.
Figure 1: (a): Plot of (16) on the surfaces z = −1.5, 0, 1.5. (b): Plot of (16) on the surface Lz = 0.
It is useful to make some considerations on the relation between equation (1), and the Frobenius integrability condition (5) for the vector field w. We have already seen that, when (5) holds, there exists local smooth functions λ and C such that w = λ∇C. Evidently, a Beltrami field can never satisfy (5)
5
unless αw2 = 0. Hence, either α = 0 or w = 0 in Ω. Both imply ∇×w = 0 in Ω. Thus, when ∇ × w 6= 0, the Frobenius integrability condition (5) is ‘dual’ to the Beltrami field condition (1), in the sense that one requires orthogonality between vector field and curl, the other their alignment.
3 Local representation of Beltrami fields
In this section we prove Theorem 1.
Proof. First we prove that (1) implies equations (8) and (9). Let vol = dx ∧ dy ∧ dz be the standard volume form of R3. To the vector field w = (wx, wy, wz) we associate the 1-form
w1 = ∗ iwvol = wxdx+ wydy + wzdz. (18)
Here i is the contraction operator and ∗ the Hodge star operator defined with respect to the Euclidean metric of R3. Next we define the 2-form ω,
ω = dw1
=
( ∂wz ∂y − ∂wy
∂z
) ∗ dx+
( ∂wx ∂z − ∂wz
∂x
) ∗ dy +
( ∂wy ∂x − ∂wx
∂y
) ∗ dz
= (∇×w)i ∗ dxi. (19)
On the other hand, any 2-form ω can be expressed in terms of an antisymmetric matrix ωij = −ωji as ω =
∑ i<j ωij dx
i ∧ dxj . In R3 this gives
ω = ωyz ∗ dx+ ωzx ∗ dy + ωxy ∗ dz. Hence ωij = εijk (∇×w)k. The rank of the 2-form ω is determined by the rank of the matrix ωij .
Since by hypothesis h 6= 0 in Ω, we have ∇×w 6= 0 in Ω. Thus the 2-form ω has rank 2 in Ω (in 3 dimensions any non-vanishing antisymmetric matrix has rank 2). Furthermore, the 2-form ω is closed:
dω = ddw1 = 0 in Ω. (20)
Therefore the hypothesis of the Lie-Darboux theorem [23, 24, 25] are verified. Hence, for every point x ∈ Ω there exists a neighborhood U ⊂ Ω of x and functions (λ,C) ∈ C∞ (U) such that
ω = dλ ∧ dC in U. (21)
This implies ω = εijkλiCj ∗ dxk and, from (19),
∇×w = ∇λ×∇C in U. (22)
6
The curl in equation (22) can be removed to give
w = ∇µ+ λ∇C in U. (23)
Since both w and λ∇C are of class C∞ (U), we have µ ∈ C∞ (U). Next recall that, by hypothesis, w is a nontrivial Beltrami field such that
h 6= 0 in Ω. On the other hand, from equation (23), we have
h = ∇µ · ∇λ×∇C 6= 0 in U. (24)
Hence, (µ, λ,C) is a coordinate system in U such that the Jacobian of the transformation (µ, λ,C) 7→ (x, y, z) is given by h. We define a second coordinate change:
` = µ sinC − λ cosC, (25a)
ψ = µ cosC + λ sinC, (25b)
θ = C. (25c)
The coordinates (`, ψ, θ) are well defined in U since the Jacobian of the coordinate change (`, ψ, θ) 7→ (µ, λ,C) is 1, i.e. h = ∇µ · ∇λ × ∇C = ∇` · ∇ψ ×∇θ. The inverse transformation is:
µ = ψ cos θ + ` sin θ, (26a)
λ = ψ sin θ − ` cos θ, (26b)
C = θ. (26c)
Substituting (26) into equation (23), we have
w = cos θ∇ψ + sin θ∇` in U. (27)
Notice that, since the Beltrami field equation (1) has not yet been used, the local representation (27) holds for any vector field w such that h 6= 0 in Ω. Rewriting equation (1) in terms of the expression for w obtained above gives
sin θ ∇ψ×∇θ+cos θ ∇θ×∇` = ∇` · ∇ψ ×∇θ
w2 (cos θ ∇ψ + sin θ ∇`) . (28)
Next, recall that tangent basis vectors (∂`, ∂ψ, ∂θ) can be expressed as
∂` = ∇ψ ×∇θ ∇` · ∇ψ ×∇θ
, ∂ψ = ∇θ ×∇`
∇` · ∇ψ ×∇θ , ∂θ =
∇`×∇ψ ∇` · ∇ψ ×∇θ
. (29)
7
Substituting these expressions in (28) we obtain
sin θ ( w2∂` −∇`
) + cos θ
( w2∂ψ −∇ψ
) = 0. (30)
Projecting (30) on the cotangent basis (∇`,∇ψ,∇θ) gives two linearly in-dependent conditions in U :
cos θ sin θ ( |∇ψ|2 − |∇`|2
) = (∇` · ∇ψ)
( cos2 θ − sin2 θ
) , (31a)
sin θ ∇` · ∇θ + cos θ ∇ψ · ∇θ = 0. (31b)
Thus, we have shown that w is a nontrivial Beltrami field provided that we can find a local coordinate system (`, ψ, θ) satisfying equation (27) and system (31). This completes the proof of the first implication. The proof of the converse statement is immediate, since system (31) guarantees that (27) is a Beltrami field, i.e. that it satisfies equation (1).
Observe that a coordinate system (`, ψ, θ) obeying system (31) is very close to an orthogonal coordinate system: at all points where either sin θ or cos θ vanish, the coordinates (`, ψ, θ) must be orthogonal. Theorem 1 has the following consequences.
Corollary 2. Let w ∈ C∞ (Ω) be a smooth vector field in a bounded domain Ω ⊂ R3 satisfying equation (1) with h = w · ∇×w 6= 0 in Ω. Then the flow generated by w has two local integral invariants θ and Lθ = ` cos θ−ψ sin θ,
w · ∇θ = w · ∇Lθ = 0 in U, (32)
where (`, ψ, θ) ∈ C∞ (U) is the local coordinate system given in Theorem 1.
Proof. The hypothesis of Theorem 1 are verified. Then w admits the local representation of equation (27). From (1) it follows that
ĥw = ∇×w = ∇θ ×∇ (` cos θ − ψ sin θ) in U. (33)
where ĥ = h/w2. Hence,
w · ∇θ = w · ∇Lθ = 0 in U. (34)
8
Observe that the level sets of the invariant θ describe the local planes where the flow generated by w lies, while the invariant Lθ = ∇θ · L̃ reflects conservation of the angular momentum-like quantity L̃ = r̃ × w, r̃ = h−1 (`∇`+ ψ∇ψ + θ∇θ), in the direction ∇θ across such planes. Note that L̃ reduces to the standard angular momentum if (`, ψ, θ) is a Carte-sian coordinate system. Furthermore, since a nontrivial Beltrami flow is locally endowed with two integral invariants, the evolution of an observable f ∈ C∞ (U) with respect to such flow can be expressed in terms of a Nambu bracket [26],
ḟ = {f, θ, Lθ} = ĥ−1∇f · ∇θ ×∇Lθ in U. (35)
If ∇ · w = 0, one has ∇ĥ · w = 0. Therefore, from (33) we see that the proportionality factor ĥ is a local function of the invariants θ and Lθ = ` cos θ − ψ sin θ alone,
ĥ = ĥ (θ, Lθ) in U. (36)
4 Construction of Beltrami fields
As a consequence of Theorem 1 it is possible to establish a method (corollary 1) to construct Beltrami fields with given proportionality coefficient ĥ. In this section we prove corollary 1 and discuss the construction procedure in detail.
Proof. First observe that by hypothesis the variables (`, ψ, θ) define an orthogonal coordinate system in Ω. Therefore |∇θ| = |α| 6= 0 in Ω, implying α 6= 0. It follows that the sign of α does not change, and we can write either α = |α| or α = − |α| in Ω. Next, note that the orthogonality condition demands that:
∇` · ∇ψ = ∇` · ∇θ = ∇ψ · ∇θ = 0 in U. (37)
Combining (37) with the assumption |∇`| = |∇ψ|, we see that system (8) is satisfied. Then, from Theorem 1, the vector field w = cos θ ∇ψ + sin θ ∇` is a Beltrami field. Let us evaluate the proportionality factor ĥ. From the definition,
ĥ = w · ∇ ×w
w2 =
∇` · ∇ψ ×∇θ |cos θ ∇ψ + sin θ ∇`|2
= σ |∇θ| = σ |α| in U, (38)
where we used the orthogonality of the local coordinate system and the assumption |∇`| = |∇ψ|. Here σ = h/ |h| is the sign of the helicity density h = w · ∇ ×w = ∇` · ∇ψ ×∇θ.
9
Consider now the vector field w∗ = sin θ ∇ψ + cos θ ∇`. We have
∇×w∗ = −h (
cos θ ∇ψ ×∇θ
h + sin θ
∇θ ×∇` h
) = −h
( cos θ
∇` |∇`|
|∂`|+ sin θ ∇ψ |∇ψ|
|∂ψ| )
= −ĥw∗ in U.
(39)
Here we used equation (29), the orthogonality condition, the assumption |∇ψ| = |∇`|, and the fact that ĥ = h/w2 = h/ |∇ψ|2. This concludes the proof.
In some cases it is useful to express the vector field w in terms of the tangent basis. From equation (30),
w = cos θ ∇ψ + sin θ ∇` = w2 (cos θ ∂ψ + sin θ ∂`) . (40)
If the coordinate system (`, ψ, θ) is orthogonal with |∇`| = |∇ψ| we have
w = cos θ ∇ψ + sin θ ∇` = |∇ψ|2 (cos θ ∂ψ + sin θ ∂`) . (41)
By using equation (41), we can evaluate the divergence ∇ ·w in the coordinate system (`, ψ, θ). First recall that
dx ∧ dy ∧ dz = h−1 d` ∧ dψ ∧ dθ. (42)
Hence, (∇ ·w) dx ∧ dy ∧ dz = Lw
( h−1 d` ∧ dψ ∧ dθ
) . (43)
It follows that
∇ ·w = ∂wx ∂x
+ ∂wy ∂y
+ ∂wz ∂z
= h
[ ∂
∂`
( w2 sin θ
h
) +
∂
∂ψ
( w2 cos θ
h
)] . (44)
If the coordinate system (`, ψ, θ) is orthogonal with |∇`| = |∇ψ| this gives
∇·w = ∂wx ∂x
+ ∂wy ∂y
+ ∂wz ∂z
= |∇ψ|2 |∇θ| [ ∂
∂`
( sin θ
|∇θ|
) +
∂
∂ψ
( cos θ
|∇θ|
)] . (45)
Corollary 1 can be slightly generalized as follows:
Proposition 1. Let (`, ψ, θ) ∈ C1 (Ω) be an orthogonal coordinate system in a bounded domain Ω ∈ R3 such that, in Ω,
|∇θ| = |α|, (46a)
|∇`| = |∇ψ| , (46b)
10
where α ∈ C (Ω). Let f ∈ C ( Ω )
be a function of the variable θ, f = f (θ). Then, the vector fields
w = cos
(∫ f dθ
) ∇ψ + sin
(∫ f dθ
) ∇`, (47a)
w∗ = sin
(∫ f dθ
) ∇ψ + cos
(∫ f dθ
) ∇`. (47b)
are Beltrami fields in Ω with proportionality factors σ|α| f and −σ|α| f respectively, where σ is the sign of the Jacobian ∇` · ∇ψ ×∇θ.
A related result is the following.
Proposition 2. Let (α, β, γ) ∈ C1 (Ω) be an orthogonal coordinate system in a bounded domain Ω ∈ R3 with |∇α| = |∇β|. Let f ∈ C1 (Ω) be a function of the variable γ, f = f (γ). Then the vector field
w = 1√
1 + f2 ∇β +
f√ 1 + f2
∇α, (48)
is a Beltrami field with proportionality factor σ |∇γ| fγ/ √
1 + f2, where σ is the sign of the Jacobian ∇α · ∇β ×∇γ.
The proof of this statement follows by performing the change of variable τ = arctan f .
Let us explain how corollary 1 is applied. Suppose that we wish to construct a Beltrami field w with given proportionality factor α 6= 0, i.e. such that ∇×w = αw. This can be accomplished by solving the following system of first order partial differential equations for the variables (`, ψ, θ) in the domain Ω:
|∇θ| = |α| , (49a)
|∇`| = |∇ψ| , (49b)
∇` · ∇ψ = 0, (49c)
∇` · ∇θ = 0, (49d)
∇ψ · ∇θ = 0. (49e)
If a solution (`, ψ, θ) ∈ C1 (Ω) is found, we set h = ∇` · ∇ψ×∇θ. Then the desired Beltrami field is w = cos θ ∇ψ + sin θ ∇` for α > 0 and h > 0 or α < 0 and h < 0, and w∗ = sin θ ∇ψ + cos θ ∇` for α < 0 and h > 0 or α > 0 and h < 0. A Beltrami field constructed in this way is endowed with the invariants θ and Lθ = ` cos θ − ψ sin θ of corollary 2 in the whole Ω.
11
Equation (49a) is the eikonal equation and can be solved by application of the method of characteristics [28]. However, notice that, even if the variable θ is known, the existence of the orthogonal coordinate system (`, ψ, θ) is not guaranteed due to the coupling among the remaining equations in system (49).
If a solenoidal Beltrami field is needed, system (49) must be supplied with an additional equation arising from the condition ∇ · w = 0. Specifically, the obtained vector fields w and w∗ will be solenoidal provided that
∇ ·w = cos θ ∆ψ + sin θ ∆` = 0 in Ω, (50a)
∇ ·w∗ = sin θ ∆ψ + cos θ ∆` = 0 in Ω. (50b)
Note that these equations are satisfied simultaneously if the coordinates ` and ψ are harmonic, i.e. if ∆` = ∆ψ = 0 in Ω.
System (49) combined with one of the equations in (50) provides a method to produce solutions of system (4), which describes steady incompressible fluid flow. This approach can be generalized to the compressible case. To see this we consider the steady compressible ideal Euler equations
(w · ∇)w = −ρ−1∇P, ∇ · (ρw) = 0 in Ω. (51)
Here ρ > 0 represents fluid density. In the following we assume that both ρ and P are smooth in Ω. Assuming a barotropic pressure P = P (ρ), we can write ∇P/ρ = ∇φ for some appropriate function φ = φ (ρ). Then system (51) reduces to
w × (∇×w) = ∇ ( φ+
w2
2
) , ∇ · (ρw) = 0 in Ω. (52)
We look for a solution of (52) in terms of a nontrivial Beltrami field. Then φ+ w2/2 = c, with c a real constant. If the function φ is invertible, we can write ρ = φ−1
( c− w2/2
) , and system (52) reduces to
w × (∇×w) = 0, ∇ log
[ φ−1
( c− w2
2
)] ·w +∇ ·w = 0 in Ω. (53)
For an ideal gas P = k ρ, with k a positive real constant. This gives φ = k log ρ and system (53) becomes
w × (∇×w) = 0, ∇ ( w2 ) ·w − 2k ∇ ·w = 0 in Ω. (54)
12
Equation (52) can be cast in an equivalent form by observing that if w is a nontrivial Beltrami field ρw = ρ ĥ−1∇×w. Then:
w× (∇×w) = 0, ∇ {[ φ−1
( c− w2
2
)] ĥ−1
} · ∇×w = 0 in Ω. (55)
In local coordinates (`, ψ, θ) the continuity equation is thus equivalent to
∇
{[ φ−1
( c− (cos θ ∇ψ + sin θ ∇`)2
2
)] (cos θ ∇ψ + sin θ ∇`)2
∇` · ∇ψ ×∇θ
} ·∇θ ×∇ (` cos θ − ψ sin θ) = 0 in U.
(56)
This shows that the existence of a nontrivial Beltrami field solution to the compressible Euler equations with barotropic pressure (53) is locally equivalent to the existence of a coordinate system (`, ψ, θ) such that equations (8), (9), and (56) hold.
Equation (56) can be simplified if the coordinate system (`, ψ, θ) is orthogonal (and thus |∇ψ| = |∇`| from system (8)) and P = k ρ is the equation of state of an ideal gas. In such case the continuity equation becomes
∇
( exp
{ k−1
( c− |∇ψ|
2
2
)} |∇θ|−1
) · ∇θ ×∇Lθ = 0. (57)
This equation is satisfied provided that there exists a function u = u (θ, Lθ) such that
|∇θ| = exp
{ −|∇ψ|
2
2k
} u. (58)
It follows that a solution to (54) with given proportionality coefficient α can be obtained by solving system (49) together with the additional condition (58) with respect to the variables (`, ψ, θ, u).
5 Examples
In this section we present a list of Beltrami fields obtained by solving system (49). Both solenoidal and non-solenoidal Beltrami fields are given.
1. Let (r, z, θ) = (√
x2 + y2, z, arctan (y/x) )
be a cylindrical coordinate
system. The vector field
w = cos z ∇ log r + sin z ∇θ, (59)
13
is a Beltrami field with proportionality factor ĥ = −1. Moreover ∇ · w = 0. The flow generated by (59) admits the invariants z and Lz = θ cos z − log r sin z.
2. Consider again the cylindrical coordinate system of the previous example. The vector field
w = sin θ ∇z + cos θ ∇r, (60)
is a Beltrami field with proportionality factor ĥ = 1/r. Moreover ∇ ·w = cos θ/r. The flow generated by (60) admits the invariants θ and Lθ = z cos θ − r sin θ.
3. Let (u, v, z) = (√ r + x,
√ r − x, z
) , r =
√ x2 + y2, be a parabolic
cylindrical coordinate system. This coordinate system is orthogonal with |∇u| = |∇v| = 1/
√ 2r. The vector field
w = cos z ∇v + sin z ∇u, (61)
is a Beltrami field with proportionality factor ĥ = 1. Moreover ∇·w = 0. The flow generated by (61) admits the invariants z and Lz = u cos z − v sin z.
4. Let (ξ, η, θ) = ( √ ρ+ z,
√ ρ− z, arctan (y/x)), ρ =
√ x2 + y2 + z2, be
a parabolic coordinate system. This coordinate system is orthogonal with |∇ξ| = |∇η| = 1/
√ 2ρ. The vector field
w = cos θ ∇η + sin θ ∇ξ, (62)
is a Beltrami field with proportionality factor ĥ = 1/r. Moreover ∇ · w = (x/η + y/ξ) /(2rρ). The flow generated by (62) admits the invariants θ and Lθ = ξ cos θ − η sin θ.
5. We want to construct a Beltrami field with proportionality factor ĥ = exp (x+ y). Solving the eikonal equation |∇θ| = exp (x+ y) gives
θ = exp (x+ y)√
2 . (63)
Next, we look for coordinates ` and ψ such that ∇` · ∇ (x+ y) = 0, ∇ψ · ∇ (x+ y) = 0, ∇` · ∇ψ = 0, and |∇`| = |∇ψ|. These conditions can be satisfied by choosing ` = z and ψ = (x− y) /
√ 2. The desired
Beltrami field is thus
w = 1√ 2
cos
[ exp (x+ y)√
2
] ∇ (x− y) + sin
[ exp (x+ y)√
2
] ∇z. (64)
14
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQE8r2bHxr8IgZulil3iop4FmI3eQwPGok2kvPaPy8MZ44-p5SXdvn-FVmbfHKvS-BmaW5tBWTlVdYl9uvDJ5vgSp__iW0ynuKsM96nLE0RuW-Pcu75gp4M2YlNuDVt2kqp79hGW9A=w1280-h711-v0?authuser=0)

Moreover ∇·w = 0. The flow generated by (64) admits the invariants θ and Lθ = ` cos θ − ψ sin θ. Figure 2 shows a plot of (64) over the integral surfaces θ = constant and Lθ = constant.
Figure 2: (a): Plot of (64) on the surfaces θ = 0.6. (b): Plot of (64) on the surface Lθ = 0.5.
6. We want to construct a Beltrami field with proportionality factor ĥ = − cos (x− y). Solving the eikonal equation |∇θ| = |cos (x− y)| gives
θ = sin (x− y)√
2 . (65)
Next, we look for coordinates ` and ψ such that∇`·∇ (x− y) = 0, ∇ψ· ∇ (x− y) = 0, ∇` ·∇ψ = 0, and |∇`| = |∇ψ|. These conditions can be satisfied by choosing ` = (z − x− y) /
√ 3 and ψ = (x+ y + 2z) /
√ 6.
The desired Beltrami field is thus
w = cos
[ sin (x− y)√
2
] ∇ ( x+ y + 2z√
6
) +
sin
[ sin (x− y)√
2
] ∇ ( z − x− y√
3
) .
(66)
Moreover ∇·w = 0. The flow generated by (66) admits the invariants θ and Lθ = ` cos θ − ψ sin θ. Figure 3 shows a plot of (66) over the integral surfaces θ = constant and Lθ = constant.
7. We want to construct a Beltrami field with proportionality factor ĥ =
15
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQEwajUGt_v8wDHSecyAWODaShd_h6FNgUgpFWRxEcm9qrxz3KuuAUuqIcI_mdePoIslam7NMOSW52kiVvpqgNtWvMPeW2EDRVHioa29wu-SiDPLkBTGe-lSqg3odjWZeXHWyrKnPA=w1280-h734-v0?authuser=0)

Figure 3: (a): Plot of (66) on the surfaces θ = 0. (b): Plot of (66) on the surface Lθ = 1.
arctan (x+ y + z). Solving |∇θ| = |arctan (x+ y + z)|, we obtain
θ =
{ (x+ y + z) arctan (x+ y + z)− 1
2 log [ 1 + (x+ y + z)2
]} / √
3.
(67) The remaining coordinates can be chosen to be ` = (x− y) /
√ 2, and
ψ = (x+ y − 2z) / √
6. The desired Beltrami field is thus
w = cos
(x+ y + z) arctan (x+ y + z)− 1 2 log
[ 1 + (x+ y + z)2
] √
3
 ∇ [ x+ y − 2z√
6
] +
sin
(x+ y + z) arctan (x+ y + z)− 1 2 log
[ 1 + (x+ y + z)2
] √
3
 ∇ ( x− y√
2
) .
(68)
Moreover ∇·w = 0. The flow generated by (68) admits the invariants θ and Lθ = ` cos θ − ψ sin θ. Figure 4 shows a plot of (68) over the integral surfaces θ = constant and Lθ = constant.
8. We look for an orthogonal coordinate system (`, ψ, θ) such that ` = ` (x, y), ψ = ψ (x, y), θ = z, and |∇ψ| = |∇`|. These conditions are
16
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQF6VtQaTBF_pPzX-6Le47KtJhdhjYj6PnAISe36hSMtkyx969mZJ1MUfKsECPDbiVfUqqBly9kUcoOb5NOm7RUj4gJEJyMB-gA8cDY4lS79tGV8QFuvocRpPzsF1l-VikMNGUy0-g=w1280-h729-v0?authuser=0)

Figure 4: (a): Plot of (68) on the surface θ = 0.2. (b): Plot of (68) on the surface Lθ = 0.5.
satisfied provided that
`y = ±ψx, `x = ∓ψy. (69)
Differentiating each equation with respect to x and y, we have
`xx + `yy = ψxx + ψyy = 0. (70)
Hence, ` and ψ must be two-dimensional harmonic functions. We set ` = ex sin y. Then, integrating (69), we find ψ = −ex cos y. It follows that the vector field
w = − cos z ∇ (ex cos y) + sin z ∇ (ex sin y) , (71)
is a Beltrami field with proportionality factor ĥ = 1. Moreover ∇·w = 0. The flow generated by (71) admits the invariants θ and Lθ = ` cos θ − ψ sin θ. Figure 5 shows a plot of (71) overt the integral surfaces θ = constant and Lθ = constant.
6 Concluding remarks
In this study, we formulated a local representation theorem for Beltrami fields by application of the Lie-Darboux theorem of differential geometry. We found that, locally, a Beltrami field has a standard form closely resembling an ABC flow with two of the parameters set to zero. In addition,
17
![image](https://lh3.googleusercontent.com/notebooklm/AKXwDQHfGOiY3r5IVYXGrOhincLOonBg594dTWxrMnn-7aDIFtV4ywyWblWoDWeRyQKAjlldT_awIHJfE3gPF0DPcZiL8_jZhnHdT1skHyeH6EF78SDfUHsYDQDyST2HYkV2m6iLiFDfDg=w1280-h722-v0?authuser=0)

Figure 5: (a): Plot of (71) on the surface θ = −1.3, 0, 1.3. (b): Plot of (71) on the surface Lθ = 0.6.
the flow generated by a Beltrami field is endowed with two local integral invariants that physically represent the plane of the flow and conservation of an angular momentum-like quantity in the direction across to the plane. The obtained local representation naturally leads to a method to construct Beltrami fields with given proportionality factor: the solution of the Bel-trami field equation (1) can be reformulated into the problem of deriving an orthogonal coordinate system satisfying certain geometric conditions. First we have to solve the eikonal equation, where the length of one of the cotangent vectors must equal the proportionality factor. The desired Beltrami field can then be obtained if the cotangent vector can be completed to an orthogonal coordinate system such that the length of the two remaining cotangent vectors are equal. Using the derived method, we constructed several Beltrami fields with constant and non-constant proportionality factors, and with zero and finite divergence.
7 Acknowledgments
The research of N. S. was supported by JSPS KAKENHI Grant No. 18J01729, and that of M. Y. by JSPS KAKENHI Grant No. 17H02860.
References
[1] Yoshida Z and Giga Y 1990 Math. Z. 204 pp 235-245
18
[2] Moffatt H K 2014 Proc. Nat. Ac. Sci. 111 10
[3] Moffatt H K 1985 J. Fluid. Mech. 159 pp 359-378
[4] Peralta-Salas D 2016 Journal of Geometric Methods in Modern Physics 13 Supp. 1 1630012
[5] Woltjer L 1958 Proc. Nat. Ac. Sci. 44 6
[6] Yoshida Z and Mahajan S M Phys. Rev. Lett. 88 9
[7] Mahajan S M and Yoshida Z Phys. Rev. Lett. 81 4863
[8] Sato N and Yoshida Z 2018 Phys. Rev. E 97 022145
[9] Etnyre J and Ghrist R 2000 Trans. Am. Math. Soc. 352 12 pp 5781-5794
[10] Enciso A, Poyato D, and Soler J 2018 Commun. Math. Phys. 360 pp 197-269
[11] Enciso A. and Peralta-Salas D. 2015 Acta Math. 214 pp 61-134
[12] Enciso A and Peralta-Salas D 2012 Ann. Math. 175 pp 345-367
[13] Dombre T, Frisch U, Greene J M, Henon M, Mehr A and Soward A M 1986 J. Fluid Mech. 167 pp 353-391
[14] Taylor J B 1986 Rev. Mod. Phys. 58 3
[15] Frankel T 2012 The Geometry of Physics, An Introduction (Cambridge: Cambridge University Press) pp 165-178
[16] Zhao X H, Kwek K H, Li J B, Huang K L 1993 SIAM J. Appl. Math. 53 1
[17] Enciso A and Peralta-Salas D 2016 Arch. Rat. Mech. Anal. 220 pp 243-260
[18] Kaiser R, Neudert M and von Wahl W 2000 Commun. Math. Phys. 211 pp 111-136
[19] Kress R 1977 J. Appl. Math. Phys. 28 pp 715-722
[20] Yoshida Z 2009 J. Math. Phys. 50 112101
[21] Yoshida Z and Morrison P J 2017 Phys. Rev. Lett. 119 244501
19
[22] Etnyre J and Ghirst R 2000 Nonlinearity 13 441
[23] de León M 1989 Methods of Differential Geometry in Analytical Me-chanics (New York: Elsevier) pp 250-253
[24] Arnold V I 1989 Mathematical Methods of Classical Mechanics (New York: Springer) pp 230-232
[25] McDuff D, Salamon D 2017 Introduction to Symplectic Topology (Ox-ford: Oxford University Press) p 110
[26] Nambu Y 1973 Phys. Rev. D 7 8
[27] Morgulis A, Yudovich V I and Zaslavsky G M 1995 Commun. Pure Appl. Math. 48 pp. 571-582
[28] Evans L C 2010 Partial Differential Equations (American Mathematical Society) pp 96-114
20