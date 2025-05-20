Codebase for topological insulators research at Princeton Lab for Topological Quantum Matter and Advanced Spectroscopy 

Goals of research (3 - fold): 
    1. ML based magnetic and TI classifier
    2. novel RL-based crystal graph structure generative modelling for TI
    3. Mech Interp for such generative models and generalizing interp methodology to other vanilla inverse design algos for materials discovery

-Abiral Shakya, 04.04.2025


-- integrate test time training with CDVAE ? --> super neat idea. could be big. - 04.12.2025

TODO Today 05.20.2025:
    re read and take notes on Dr. Hasan review paper

    Scrap solely RL, it didnt work too great + not enough theory to work off of for successful policy IMO, rather read more into transformer based materials discovery, probably SSL is way to go

    Begin scripting based off of materials discovery papers using transformers

    Further fine tuning on multi task classifier


2D integer quantum Hall state, surface / edge state of TI lead to conducting state. gapless states exist unlike ordinary inssulator. 
Topological Order: characterize intricately correlated fractional quantum hall states, require many body approach. 
Topological considerations also apply to integer quantum Hall State, decribed properly withijn single particle quantum mech

Single particle energy gap exists --> electron-electron interactions do not moodify state in essential way 
TODO: read further literature about experiment, theory, and simulation regarding electron electron interaction effect on *successfully* characterized TIs

understood within band theory of solids. 

topological band theory:
-insulating state: crystal momentum k. Bloch states defiend in single u nit cell of the crystal, are eigenstates of Bloch Hamiltonian and eigenvalue define energy band dthat collectigely form the band structure. 
topological equivalance between dif insulating state (tuning Hamiltonian to interpolate cont between the two without closign energy gap)
    -equate state with different numbers of trivial core bands, then all conventional insulators are equivalent
        (also equivalent to vacuum which has conduction band (electrons) and valence bond (positrons))

counterexamples of states of matteer that are not topologically equivalent to vaccuum:
    -integer quantum hall state, electrons confined to 2D placed in trong magnetic field. quantization of electrons circular orbits with cyclotron frequency omega_c lead to q uantized Landau Level.  
        N Landau level filled & rest empty energy gap eperates occupied emtpy states just as insulator
        Electric Field causes cyclotron orbits to drift leading to hall current characterized by quantized Hall conductivity 
        band structure identical to trad insulator 

TKNN invariant
    2D band structure consists of mapping from crystal momentum k to bloch hamiltonian. gapped band structures classified topologically by considering equivalence classess of H(k) that can be cont deformed dinto one another without closing the energy gap. 
        Chern invariant n (integer n) distinguishes such classes
            chern invariant is total Berry flux in Brilloiun Zone 
            
    N = n

Graphene 
    2D form of carbon, conduction band and valence band touch each other at 2 distinct points in Brilloiun Zone. near thosse poitns electronic dispersion resemble linear dispersion of malsless reletivitic particles


Chiral edge mode, interface between quantum hall state and an insulator

electronic states responsible for skipping motion electrons execute as their cyclotron orbits bounce off the edge
    chiral in the sense that they propogate in one direction onl along the edge. 

chiral edge states in quantum hall effect seen explicitly by solving haldane model in semi infintie geometry with an edge at y = 0

