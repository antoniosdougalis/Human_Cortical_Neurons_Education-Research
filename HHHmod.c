/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__HHHmod
#define _nrn_initial _nrn_initial__HHHmod
#define nrn_cur _nrn_cur__HHHmod
#define _nrn_current _nrn_current__HHHmod
#define nrn_jacob _nrn_jacob__HHHmod
#define nrn_state _nrn_state__HHHmod
#define _net_receive _net_receive__HHHmod 
#define states states__HHHmod 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gnabar _p[0]
#define gnabar_columnindex 0
#define gkbar _p[1]
#define gkbar_columnindex 1
#define glbar _p[2]
#define glbar_columnindex 2
#define el _p[3]
#define el_columnindex 3
#define ina _p[4]
#define ina_columnindex 4
#define ik _p[5]
#define ik_columnindex 5
#define il _p[6]
#define il_columnindex 6
#define minf _p[7]
#define minf_columnindex 7
#define hinf _p[8]
#define hinf_columnindex 8
#define ninf _p[9]
#define ninf_columnindex 9
#define tau_m _p[10]
#define tau_m_columnindex 10
#define tau_h _p[11]
#define tau_h_columnindex 11
#define tau_n _p[12]
#define tau_n_columnindex 12
#define m _p[13]
#define m_columnindex 13
#define h _p[14]
#define h_columnindex 14
#define n _p[15]
#define n_columnindex 15
#define ena _p[16]
#define ena_columnindex 16
#define ek _p[17]
#define ek_columnindex 17
#define Dm _p[18]
#define Dm_columnindex 18
#define Dh _p[19]
#define Dh_columnindex 19
#define Dn _p[20]
#define Dn_columnindex 20
#define v _p[21]
#define v_columnindex 21
#define _g _p[22]
#define _g_columnindex 22
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
#define _ion_ek	*_ppvar[3]._pval
#define _ion_ik	*_ppvar[4]._pval
#define _ion_dikdv	*_ppvar[5]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_boltz(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_HHHmod", _hoc_setdata,
 "boltz_HHHmod", _hoc_boltz,
 0, 0
};
#define boltz boltz_HHHmod
 extern double boltz( _threadargsprotocomma_ double , double , double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gnabar_HHHmod", "S/cm2",
 "gkbar_HHHmod", "S/cm2",
 "glbar_HHHmod", "S/cm2",
 "el_HHHmod", "mV",
 "ina_HHHmod", "mA/cm2",
 "ik_HHHmod", "mA/cm2",
 "il_HHHmod", "mA/cm2",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"HHHmod",
 "gnabar_HHHmod",
 "gkbar_HHHmod",
 "glbar_HHHmod",
 "el_HHHmod",
 0,
 "ina_HHHmod",
 "ik_HHHmod",
 "il_HHHmod",
 "minf_HHHmod",
 "hinf_HHHmod",
 "ninf_HHHmod",
 "tau_m_HHHmod",
 "tau_h_HHHmod",
 "tau_n_HHHmod",
 0,
 "m_HHHmod",
 "h_HHHmod",
 "n_HHHmod",
 0,
 0};
 static Symbol* _na_sym;
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 23, _prop);
 	/*initialize range parameters*/
 	gnabar = 0.05;
 	gkbar = 0.01;
 	glbar = 8e-05;
 	el = -32;
 	_prop->param = _p;
 	_prop->param_size = 23;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[3]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[4]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[5]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _HHHmod_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	ion_reg("k", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 23, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 HHHmod HHHmod.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Fast Sodium, potassium delayed rectifier and leak currents";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[3], _dlist1[3];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   double _lminf , _lhinf , _lninf , _ltau_m , _ltau_h , _ltau_n ;
 _lminf = boltz ( _threadargscomma_ v , - 41.0 , 4.5 ) ;
   _lhinf = boltz ( _threadargscomma_ v , - 70.0 , - 6.5 ) ;
   _lninf = boltz ( _threadargscomma_ v , - 25.0 , 12.0 ) ;
   _ltau_m = boltz ( _threadargscomma_ v , - 45.0 , - 1.5 ) - boltz ( _threadargscomma_ v , - 65.0 , - 0.5 ) + 0.04 ;
   _ltau_h = 56.0 * boltz ( _threadargscomma_ v , - 39.0 , - 4.5 ) - 56.0 * boltz ( _threadargscomma_ v , - 59.0 , - 2.0 ) + 1.0 ;
   _ltau_n = 300.0 * boltz ( _threadargscomma_ v , - 38.4 , - 6.9 ) ;
   Dm = ( _lminf - m ) / _ltau_m ;
   Dh = ( _lhinf - h ) / _ltau_h ;
   Dn = ( _lninf - n ) / _ltau_n ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 double _lminf , _lhinf , _lninf , _ltau_m , _ltau_h , _ltau_n ;
 _lminf = boltz ( _threadargscomma_ v , - 41.0 , 4.5 ) ;
 _lhinf = boltz ( _threadargscomma_ v , - 70.0 , - 6.5 ) ;
 _lninf = boltz ( _threadargscomma_ v , - 25.0 , 12.0 ) ;
 _ltau_m = boltz ( _threadargscomma_ v , - 45.0 , - 1.5 ) - boltz ( _threadargscomma_ v , - 65.0 , - 0.5 ) + 0.04 ;
 _ltau_h = 56.0 * boltz ( _threadargscomma_ v , - 39.0 , - 4.5 ) - 56.0 * boltz ( _threadargscomma_ v , - 59.0 , - 2.0 ) + 1.0 ;
 _ltau_n = 300.0 * boltz ( _threadargscomma_ v , - 38.4 , - 6.9 ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / _ltau_m )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / _ltau_h )) ;
 Dn = Dn  / (1. - dt*( ( ( ( - 1.0 ) ) ) / _ltau_n )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   double _lminf , _lhinf , _lninf , _ltau_m , _ltau_h , _ltau_n ;
 _lminf = boltz ( _threadargscomma_ v , - 41.0 , 4.5 ) ;
   _lhinf = boltz ( _threadargscomma_ v , - 70.0 , - 6.5 ) ;
   _lninf = boltz ( _threadargscomma_ v , - 25.0 , 12.0 ) ;
   _ltau_m = boltz ( _threadargscomma_ v , - 45.0 , - 1.5 ) - boltz ( _threadargscomma_ v , - 65.0 , - 0.5 ) + 0.04 ;
   _ltau_h = 56.0 * boltz ( _threadargscomma_ v , - 39.0 , - 4.5 ) - 56.0 * boltz ( _threadargscomma_ v , - 59.0 , - 2.0 ) + 1.0 ;
   _ltau_n = 300.0 * boltz ( _threadargscomma_ v , - 38.4 , - 6.9 ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / _ltau_m)))*(- ( ( ( _lminf ) ) / _ltau_m ) / ( ( ( ( - 1.0 ) ) ) / _ltau_m ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / _ltau_h)))*(- ( ( ( _lhinf ) ) / _ltau_h ) / ( ( ( ( - 1.0 ) ) ) / _ltau_h ) - h) ;
    n = n + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / _ltau_n)))*(- ( ( ( _lninf ) ) / _ltau_n ) / ( ( ( ( - 1.0 ) ) ) / _ltau_n ) - n) ;
   }
  return 0;
}
 
double boltz ( _threadargsprotocomma_ double _lx , double _ly , double _lz ) {
   double _lboltz;
 _lboltz = 1.0 / ( 1.0 + exp ( - ( _lx - _ly ) / _lz ) ) ;
   
return _lboltz;
 }
 
static void _hoc_boltz(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  boltz ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
  ek = _ion_ek;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
   }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 3; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
   nrn_update_ion_pointer(_k_sym, _ppvar, 3, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 4, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 5, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
  n = n0;
 {
   m = minf ;
   h = hinf ;
   n = ninf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
  }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ina = gnabar * m * m * m * h * ( v - ena ) ;
   ik = gkbar * n * n * n * n * ( v - ek ) ;
   il = glbar * ( v - el ) ;
   }
 _current += ina;
 _current += ik;
 _current += il;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
  ek = _ion_ek;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
 double _dina;
  _dina = ina;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
  _ion_ik += ik ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ena = _ion_ena;
  ek = _ion_ek;
 {   states(_p, _ppvar, _thread, _nt);
  }  }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
 _slist1[2] = n_columnindex;  _dlist1[2] = Dn_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "HHHmod.mod";
static const char* nmodl_file_text = 
  "TITLE  Fast Sodium, potassium delayed rectifier and leak currents\n"
  "\n"
  "COMMENT\n"
  "written and modified by Antonios Dougalis to fit data from Komendantov et al., 2004 J Neurophysiol, 10 Aug 2015, Ulm\n"
  "ENDCOMMENT\n"
  "\n"
  "INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "UNITS {\n"
  "        (S) = (siemens)\n"
  "        (mA) = (milliamp)\n"
  "        (mV) = (millivolt)\n"
  "}\n"
  " \n"
  "NEURON {\n"
  "        SUFFIX HHHmod\n"
  "        USEION na READ ena WRITE ina\n"
  "        USEION k READ ek WRITE ik\n"
  "		NONSPECIFIC_CURRENT il \n"
  "        RANGE gnabar,gkbar,glbar,ina,ik,il,ena,ek,el\n"
  "        RANGE minf,hinf,ninf\n"
  "		RANGE tau_m,tau_h,tau_n\n"
  "}\n"
  " \n"
  "PARAMETER {\n"
  "        v   (mV)\n"
  "        dt  (ms)\n"
  "		gnabar  = 0.050 (S/cm2)\n"
  "        gkbar = 0.01 (S/cm2)\n"
  "        glbar  = 0.00008  (S/cm2)\n"
  "        ena = 50 (mV)\n"
  "		ek  = -73.0  (mV)\n"
  "		el = -32 (mV)\n"
  "              \n"
  "}\n"
  " \n"
  "STATE {\n"
  "        m h n\n"
  "}\n"
  " \n"
  "ASSIGNED {\n"
  "        ina (mA/cm2)\n"
  "        ik (mA/cm2)\n"
  "        il (mA/cm2)\n"
  "        minf\n"
  "		hinf \n"
  "		ninf \n"
  "	    tau_m\n"
  "		tau_h\n"
  "		tau_n\n"
  "}\n"
  " \n"
  "BREAKPOINT {\n"
  "        SOLVE states METHOD cnexp\n"
  "        ina = gnabar*m*m*m*h*(v - ena)\n"
  "        ik = gkbar*n*n*n*n*(v - ek)      \n"
  "        il = glbar*(v - el)      \n"
  "}\n"
  " \n"
  "UNITSOFF\n"
  "\n"
  "INITIAL {\n"
  "        m = minf\n"
  "        h = hinf\n"
  "        n = ninf\n"
  "}\n"
  "\n"
  "DERIVATIVE states { \n"
  "        LOCAL minf,hinf,ninf,tau_m,tau_h,tau_n\n"
  "        minf = boltz(v,-41.0,4.5)\n"
  "        hinf = boltz(v,-70.0,-6.5)\n"
  "        ninf = boltz(v,-25,12.0)\n"
  "        tau_m = boltz(v,-45.0,-1.5) - boltz(v,-65.0,-0.5) +0.04\n"
  "        tau_h = 56.0*boltz(v,-39,-4.5) - 56.0*boltz(v,-59,-2.0) +1.0\n"
  "        tau_n = 300*boltz(v,-38.4,-6.9)\n"
  "		m' = (minf-m)/tau_m\n"
  "        h' = (hinf-h)/tau_h\n"
  "        n' = (ninf-n)/tau_n\n"
  "}\n"
  " \n"
  "FUNCTION boltz(x,y,z) {\n"
  "                boltz = 1/(1 + exp(-(x - y)/z))\n"
  "}\n"
  " \n"
  "UNITSON\n"
  ;
#endif
