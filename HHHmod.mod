TITLE  Fast Sodium, potassium delayed rectifier and leak currents

COMMENT
written and modified by Antonios Dougalis to fit data from Komendantov et al., 2004 J Neurophysiol, 10 Aug 2015, Ulm
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

UNITS {
        (S) = (siemens)
        (mA) = (milliamp)
        (mV) = (millivolt)
}
 
NEURON {
        SUFFIX HHHmod
        USEION na READ ena WRITE ina
        USEION k READ ek WRITE ik
		NONSPECIFIC_CURRENT il 
        RANGE gnabar,gkbar,glbar,ina,ik,il,ena,ek,el
        RANGE minf,hinf,ninf
		RANGE tau_m,tau_h,tau_n
}
 
PARAMETER {
        v   (mV)
        dt  (ms)
		gnabar  = 0.050 (S/cm2)
        gkbar = 0.01 (S/cm2)
        glbar  = 0.00008  (S/cm2)
        ena = 50 (mV)
		ek  = -73.0  (mV)
		el = -32 (mV)
              
}
 
STATE {
        m h n
}
 
ASSIGNED {
        ina (mA/cm2)
        ik (mA/cm2)
        il (mA/cm2)
        minf
		hinf 
		ninf 
	    tau_m
		tau_h
		tau_n
}
 
BREAKPOINT {
        SOLVE states METHOD cnexp
        ina = gnabar*m*m*m*h*(v - ena)
        ik = gkbar*n*n*n*n*(v - ek)      
        il = glbar*(v - el)      
}
 
UNITSOFF

INITIAL {
        m = minf
        h = hinf
        n = ninf
}

DERIVATIVE states { 
        LOCAL minf,hinf,ninf,tau_m,tau_h,tau_n
        minf = boltz(v,-41.0,4.5)
        hinf = boltz(v,-70.0,-6.5)
        ninf = boltz(v,-25,12.0)
        tau_m = boltz(v,-45.0,-1.5) - boltz(v,-65.0,-0.5) +0.04
        tau_h = 56.0*boltz(v,-39,-4.5) - 56.0*boltz(v,-59,-2.0) +1.0
        tau_n = 300*boltz(v,-38.4,-6.9)
		m' = (minf-m)/tau_m
        h' = (hinf-h)/tau_h
        n' = (ninf-n)/tau_n
}
 
FUNCTION boltz(x,y,z) {
                boltz = 1/(1 + exp(-(x - y)/z))
}
 
UNITSON
