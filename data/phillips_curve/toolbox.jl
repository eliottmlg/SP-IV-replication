using ProgressMeter
using StatsBase, LinearAlgebra, Statistics, FFTW
using Printf
using MAT
using PyPlot
using Random

############################################################
#          Defining useful structures
############################################################

struct obj_minnesota_priors
 	th0
	sigma0
	isigma0
	gamma0
	t0
end

struct obj_gibbs
	coefs
	vol
end

struct var_results
    y
    x
    coefs
    stdcoefs
    vcovcoefs
    vcov
    Yfit
    resid
    SSR
    nobs
    nvar
    lags
    mx
    aic
    bic
    hq
end

struct state_space_model
  mx
  my
  me
  sig
end

struct factor_model
  mx
  my
  me
  mu
  sig
end

struct irf
  x
  y
end

############################################################
#          Matlab like functions and other utilities
############################################################
function basic_ols(y::Vector,x::Matrix)
    N=length(x)
    X=hcat(x,ones(N))
    return X\y
end

function basic_ols(y::Vector,x::Vector)
    N=length(x)
    X=hcat(x,ones(N))
    return X\y
end

function eye(n::Int64)
  return diagm(0=>ones(n))
end

function eye(n1::Int64,n2::Int64)
  return hcat(diagm(0=>ones(n1)),zeros(n1,n2-n1))
end

function chol(A)
  C=cholesky(Hermitian(A))
  return C.U;
end

function linspace(xmin,xmax,nb::Int64)
  return collect(range(xmin,stop=xmax,length=nb))
end

function eig(A)
  D,P=eigen(A)
  return D,P
end

function indmax(x)
  return argmax(x)
end

function fprintf(head,X)
    print(head)
    for i=1:length(X)
        @printf("& %5.2f ",X[i])
    end
    println("\\\\")
end


function printvar(head,X::Matrix,order::Vector;level=16)
    nx=size(X,2)
    Z=vcat([percentile(X[:,i],[level 50 100-level]) for i in 1:nx]...)
    print(head)
    for i in order
        @printf("& %5.2f ",Z[i,2])
    end
    println("\\\\")
    for i in order
        @printf("& {\\scriptsize[%5.2f,%5.2f]} ",Z[i,1],Z[i,3])
    end
    println("\\\\")
end

function printvar(head,X1::Matrix,X2::Matrix,order1::Vector,order2::Vector;level=16)
    nx1=size(X1,2)
    nx2=size(X2,2)
    Z1=vcat([percentile(X1[:,i],[level 50 100-level]) for i in 1:nx1]...)
    Z2=vcat([percentile(X2[:,i],[level 50 100-level]) for i in 1:nx2]...)
    print(head)
    for i in order1
        @printf("& %5.2f ",Z1[i,2])
    end
    print("&")
    for i in order2
        @printf("& %5.2f ",Z2[i,2])
    end
    println("\\\\")
    for i in order1
        @printf("& {\\scriptsize[%5.2f,%5.2f]} ",Z1[i,1],Z1[i,3])
    end
    print("&")
    for i in order2
        @printf("& {\\scriptsize[%5.2f,%5.2f]} ",Z2[i,1],Z2[i,3])
    end
    println("\\\\")
end

############################################################
#                  VAR estimation functions
############################################################

function buildmat(X::Matrix,lags::Int64,constant=true)
    T,k=size(X)
    XX=zeros(T-lags,lags*k)
    for i=1:lags
        XX[:,(i-1)*k+1:i*k]=X[lags+1-i:T-i,:];
    end
    YY=X[lags+1:T,:];
    if constant
        XX=[XX ones(T-lags,1)];
    end
    return YY,XX
end

function varest(X,lags;constant=true)
    (nobs,nvar) = size(X);
    (YY,XX)     = buildmat(X,lags,constant);
    T           = size(XX,1);
    coefs       = XX\YY;
    ncoefs      = size(coefs,1);
    XXi         = (XX'*XX)\eye(ncoefs);
    Yfit        = XX*coefs;
    resid       = YY-Yfit;
    SSR         = resid'*resid;
    vcov        = SSR/(T-lags);
    ldet        = log(det(cov(resid)));
    nbc         = length(coefs)
    aic         = ldet+2*nbc/nobs;
    bic         = ldet+nbc*log(nobs)/nobs;
    hq          = ldet+2*nbc*log(log(nobs))/nobs;
    VX          = kron(vcov,XXi);
    SX          = reshape(sqrt.(diag(VX)),ncoefs,nvar);
    MX          = [coefs[1:lags*nvar,:]';eye((lags-1)*nvar,lags*nvar)];

    return var_results(YY,XX,coefs,SX,VX,vcov,Yfit,resid,SSR,nobs,nvar,lags,MX,aic,bic,hq);
end

############################################################
#           Bayesian VAR estimation functions
############################################################

function minnesota_priors(data::Array{Float64,2},AR,lags::Int64,param=[0.2,0.5,2.0,1e5])
	T,N 	= size(data);
    lb1     = param[1];
    lb2     = param[2];
    lb3     = param[3];
    lb4     = param[4];
	if ~isempty(AR)
    	rho = AR[:];
	else
    	rho = ones(N);
	end

	se      = zeros(N);
	for i in 1:N
	    y   = data[2:T,i];
	    x   = [ones(T-1,1) data[1:T-1,i]];
	    b   = x\y;
	    u   = y-x*b;
	    se[i] = sqrt(dot(u,u)/(size(y,1)-2));
	end
	H   = zeros(N*lags+1,N);
	TH  = zeros(N*lags+1,N);
	for i in 1:N
    	k       = 1;
	    for l in 1:lags
	        for j in 1:N;
	            if j==i
	                tmp     = lb1/l^lb3;
	                H[k,i]  = tmp*tmp;
	                if l==1;
	                    TH[k,i] = rho[i];
	                end
	            else
	                tmp     = se[i]*lb1*lb2/(se[j]*l^lb3);
	                H[k,i]  = tmp*tmp;
	            end
	            k   = k+1;
	        end
	    end
	    tmp     = se[i]*lb4;
	    H[k,i]  = tmp*tmp;
	    TH[k,i] = 0;
	end
	TH              = TH[:];
	H               = diagm(0 => H[:]);
	IH              = H\eye(N*(N*lags+1));
	return obj_minnesota_priors(TH,H,IH,eye(N),N+1)
end


function gibbs_var(resvar,priors,Ndraws,Nburn;seed=[])
    if !isempty(seed)
        Random.seed!(seed)
    end
	ISIG0   = priors.isigma0;
	TH0     = priors.th0;
	T0      = priors.t0;
	GAMMA0  = priors.gamma0;
	BOLS    = resvar.coefs;
	SIGMA   = resvar.vcov;
	X       = resvar.x;
	Y       = resvar.y;
	XX      = X'*X;
	lags    = resvar.lags;
	T       = resvar.nobs;
	nvar    = resvar.nvar;
	ncoef   = size(BOLS,1);

	# Gibbs' sampler
	ncoef   = size(TH0,1);
	Coef    = zeros(Ndraws-Nburn,ncoef);
	Vol     = zeros(Ndraws-Nburn,nvar*nvar);
	k       = 1;
	pbar 	= Progress(Ndraws,1,"Progress ... ",80)
	for i=1:Ndraws
	    # 1) Draw coefficients conditional on volatility
	    isig    = inv(chol(Hermitian(SIGMA)));
	    ISIG    = isig*isig';
	    tmp     = (ISIG0+kron(ISIG,XX));
      tmp     = (tmp+tmp')/2.0
	    itmp    = inv(chol(tmp));
	    Vstar   = itmp*itmp';
      Vstar   = (Vstar+Vstar')/2.0

	    Mstar   = Vstar*(ISIG0*TH0+kron(ISIG,XX)*BOLS[:]);
	    chk     = 0
	    TH1     = Float64[]
	    while chk==0 # This test insures stability of the VAR
	    	TH1     = Mstar+chol(Vstar)'*randn(nvar*(lags*nvar+1));
	    	TH1     = reshape(TH1,nvar*lags+1,nvar);
	    	MX      = [TH1[1:lags*nvar,:]';eye((lags-1)*nvar,lags*nvar)];
    		lbmax	= maximum(abs.(eigvals(MX)))
	    	if lbmax<1.0
	    		chk = 1
	    	end
		end
	    # 2) Draw volatility conditional on new coefficients
	    resid   = (Y-X*TH1)-broadcast(*,mean(Y-X*TH1,dims=1),ones(T-lags,1));
	    T1      = T0+T;
	    GAMMA1  = GAMMA0+resid'*resid;
	    IG1     = GAMMA1\eye(nvar);
      IG1     = (IG1+IG1')/2.0
	    z       = randn(T1,nvar)*chol(IG1);
	    SIGMA   = (z'*z)\eye(nvar);
	    # 3) Keeps the draw if i>Nburn
	    if i>Nburn
	        Coef[k,:]   = TH1[:];
	        Vol[k,:]    = SIGMA[:];
	        k           = k+1;
	    end
        update!(pbar,i)
	end
	return obj_gibbs(Coef,Vol)
end

############################################################
#         Functions pertaining to our project
############################################################

function funcfdq(res::state_space_model,S,wmin,wmax,idx,weight;grid=1024,all=false)
    # Computes the impulse vector for a specific set of variables
    freqs   = linspace(0.0,2*pi,grid)
    F       = zeros(size(freqs));
    F[(freqs.>=wmin).&(freqs.<=wmax)]  .= 1;
    F[(freqs.>=(2*pi-wmax)).&(freqs.<=(2*pi-wmin))]  .= 1;
    ne      = size(S,2)
    MX      = res.mx;
    nx      = size(MX,1);
    MY      = res.my;
    S       = Array(S)
    ME      = [S[:,1:ne];zeros(nx-ne,ne)];
    zi      = exp.(-im*freqs);
    r2pi    = 1/(2*pi);
    nidx    = length(idx);
    VD      = zeros(ne,ne);
    for i in 1:nidx
        sp      = complex(zeros(grid,1));
        sp2     = complex(zeros(grid,ne*ne));

        for gp in 1:grid;
            if F[gp]==1
                fom     = MY[idx[i],:]'*((eye(nx)-MX*zi[gp])\ME);
                tmp     = r2pi*(fom*fom');
                tmp     = F[gp]*tmp;
                sp[gp]  = tmp;
                tmp     = r2pi*(fom'*fom);
                tmp     = F[gp]*tmp;
                sp2[gp,:] = tmp[:]';
            end
        end;
        sp[isnan.(sp)]   .= 0.0;
        sp2[isnan.(sp2)] .= 0.0;
        VTtmp           = 2*pi*real(ifft(sp,1))
        VDtmp           = 2*pi*real(ifft(sp2,1));
        VD              = VD+weight[i]*reshape(VDtmp[1,:]/VTtmp[1],ne,ne);
    end
    D,P   = eig(VD);
    i     = indmax(abs.(D));
    if all
        return D,P
    else
        return real(P[:,i]);
    end
end

function funcfdq_constrained(res::state_space_model,S,C,wmin,wmax,idx,weight,grid=1024)
    freqs   = linspace(0.0,2*pi,grid);
    F       = zeros(size(freqs));
    F[(freqs.>=wmin).&(freqs.<=wmax)]  .= 1;
    F[(freqs.>=(2*pi-wmax)).&(freqs.<=(2*pi-wmin))]  .= 1;
    ne      = size(S,2)
    MX      = res.mx;
    nx      = size(MX,1);
    MY      = res.my;
    S       = Array(S)
    ME      = [S[:,1:ne];zeros(nx-ne,ne)];
    zi      = exp.(-im*freqs);
    r2pi    = 1/(2*pi);
    nidx    = length(idx);
	VD      = zeros(ne,ne);
    for i in 1:nidx
        sp      = complex(zeros(grid,1));
        sp2     = complex(zeros(grid,(ne)*(ne)));
        for gp in 1:grid;
            if F[gp]==1
                fom     = MY[idx[i],:]'*((eye(nx)-MX*zi[gp])\ME);
                tmp     = r2pi*(fom*fom');
                tmp     = F[gp]*tmp;
                sp[gp]  = tmp
                tmp     = r2pi*(fom'*fom);
                tmp     = F[gp]*tmp;
                sp2[gp,:] = tmp[:]';
            end
        end;

        sp[isnan.(sp)]   .= 0;
        sp2[isnan.(sp2)] .= 0;
        VTtmp           = 2*pi*real(ifft(sp,1));
        VDtmp           = 2*pi*real(ifft(sp2,1));
        VD              = VD+weight[i]*reshape(VDtmp[1,:]/VTtmp[1],ne,ne);
    end
    Z     = nullspace(C)
    D,P   = eig(Z'*VD*Z);
    P     = Z*P
    i     = indmax(abs.(D));
    return real(P[:,i]);
end


function functdq(q0,q1,Vtmp1,idx,maxhor;zero_impact=false)
    q0    = q0+1 # Here we add 1 because the first index is the impact effect
    q1    = q1+1
    smpl  = (idx[1]-1)*maxhor+1:idx[1]*maxhor
    Itmp  = Vtmp1[smpl,:]
    n     = size(Vtmp1,2)
    if q1>q0
        V     = zeros(n,n);
        for j=q0:q1
            V0= zeros(n,n);
            V1= 0
            for k=1:j
                V0 = V0+Itmp[k,:]*Itmp[k,:]'
                V1 = V1+Itmp[k,:]'*Itmp[k,:]
            end
        end
    else
        V0= zeros(n,n)
        V1= 0
        for k=1:q0
            V0 = V0+Itmp[k,:]*Itmp[k,:]'
            V1 = V1+Itmp[k,:]'*Itmp[k,:]
        end
    end
    V               = V0/V1;
    if zero_impact
        D,P         = eig(V[2:n,2:n])
        i           = indmax(abs.(D))
        Q           = vcat(0,P[:,i])
    else
        D,P         = eig(V)
        i           = indmax(abs.(D))
        Q           = P[:,i]
    end
    return Q
end

function share_trans(res::state_space_model,S,Q;grid=128)
    # Computes the variance decomposition

    wmin    = 2*[pi/32,0];
    wmax    = 2*[pi/6,pi/80];

    # freqs   = linspace(0.0,2*pi,grid);
    freqs   = linspace(1e-6,2*pi,grid);
    Fsr     = zeros(size(freqs));
    Flr     = zeros(size(freqs));

    Fsr[(freqs.>=wmin[1]).&(freqs.<=wmax[1])].= 1;
    Fsr[(freqs.>=(2*pi-wmax[1])).&(freqs.<=(2*pi-wmin[1]))].= 1;
    Flr[(freqs.>=wmin[2]).&(freqs.<=wmax[2])].= 1;
    Flr[(freqs.>=(2*pi-wmax[2])).&(freqs.<=(2*pi-wmin[2]))].= 1;

    MX      = res.mx;
    nx      = size(MX,1);
    ne      = size(S,2);
    MY      = res.my;
    ME0     = [S;zeros(nx-ne,ne)];
    ME2     = [S*Q;zeros(nx-ne,1)];
    zi      = exp.(-im*freqs);
    r2pi    = 1/(2*pi);
    ny      = size(MY,1);
    SVTsr   = zeros(1,ny);
    SVTlr   = zeros(1,ny);

    for i in 1:ny
        spsr0     = complex(zeros(grid,1));
        splr0     = complex(zeros(grid,1));
        spsr2     = complex(zeros(grid,1));
        splr2     = complex(zeros(grid,1));
        for gp in 1:grid;
            if Fsr[gp]==1
                fom0        = MY[i,:]'*((eye(nx)-MX*zi[gp])\ME0);
                fom2        = MY[i,:]'*((eye(nx)-MX*zi[gp])\ME2);
                spsr0[gp]   = r2pi*(fom0*fom0')
                spsr2[gp]   = r2pi*(fom2*fom2');
            end
            if Flr[gp]==1
                fom0        = MY[i,:]'*((eye(nx)-MX*zi[gp])\ME0);
                fom2        = MY[i,:]'*((eye(nx)-MX*zi[gp])\ME2);
                splr2[gp]   = r2pi*(fom2*fom2');
                splr0[gp]   = r2pi*(fom0*fom0');
            end;
        end
        spsr0[isnan.(spsr0)].=0;
        spsr2[isnan.(spsr2)].=0;
        splr0[isnan.(splr0)].=0;
        splr2[isnan.(splr2)].=0;

        V0tmp       = 2*pi*real(ifft(spsr0,1));
        V2tmp       = 2*pi*real(ifft(spsr2,1));
        SVTsr[i]    = 100*V2tmp[1]/V0tmp[1]; # Share in total Volatility
        V0tmp       = 2*pi*real(ifft(splr0,1));
        V2tmp       = 2*pi*real(ifft(splr2,1));
        SVTlr[i]    = 100*V2tmp[1]/V0tmp[1]; # Share in total Volatility
    end
    return SVTsr,SVTlr
end

function vdec(IRFe,Q,nrep)
    N,k=size(IRFe)
    col=Int(N/nrep)
    IRFq=reshape(IRFe*Q,nrep,col)
    Vnum=cumsum(IRFq.*IRFq,dims=1)
    Vden=zeros(nrep,col)
    for j=1:k
        tmp0=reshape(IRFe[:,j],nrep,col)
        Vden=Vden+cumsum(tmp0.*tmp0,dims=1)
    end
    return Vnum./Vden
end

function comp_irf(res::state_space_model,k,nrep)
  MX=res.mx
  MY=res.my
  ME=res.me
  nx,ne=size(ME)
  I      = eye(ne)
  X      = zeros(nx,nrep)
  X[:,1] = ME*I[:,k]
  for t in 2:nrep
    X[:,t] = MX*X[:,t-1]
  end
  Y=(MY*X)'
  X=X'
  return irf(X,Y)
end;
;


function bpass(X,pl,pu;undrift=true)
    #
    # Julia COMMAND FOR BAND PASS FILTER: fX = bpass(X,pl,pu; undrift=true)
    #
    # This is a Julia port of Eduard Pelz's program that filters time series data using an approximation to the
    # band pass filter as discussed in the paper "The Band Pass Filter" by Lawrence J.
    # Christiano and Terry J. Fitzgerald (1999).
    #
    # Required Inputs:
    #   X     - series of data (T x 1)
    #   pl    - minimum period of oscillation of desired component
    #   pu    - maximum period of oscillation of desired component (2<=pl<pu<infinity).
    #   undrift to get rid of a trend (default is true)
    #
    # Output:
    #   fX - matrix (T x 1) containing filtered data
    #
    # Examples:
    #   Quarterly data: pl=6, pu=32 returns component with periods between 1.5 and 8 yrs.
    #   Monthly data:   pl=2, pu=24 returns component with all periods less than 2 yrs.
    #
    #  Note:  When feasible, we recommend dropping 2 years of data from the beginning
    #         and end of the filtered data series.  These data are relatively poorly estimated.
    #
    # ===============================================================================
    #  This program contains only the default filter recommended in Christiano and
    #  Fitzgerald (1999). This program was originally written by Eduard Pelz for Matlab
    #  and ported to Julia by F. Collard. Any errors are my own and not those of the authors
    #  ===============================================================================
    #
    #  Version Date: 2/11/00 (Please notify Eduard Pelz at eduard.pelz@clev.frb.org or
    #  (216)579-2063 if you encounter bugs).
    #  This Version: 03/18/16.
    #

    if pu <= pl
        error(" (bpass): pu must be larger than pl")
    end
    if pl < 2
        println("bpass: pl less than 2 , reset to 2")
        pl = 2;
    end

    Tmp = size(X);
    T   = Tmp[1];
    if length(Tmp)>1
        nvars=Tmp[2]
        id=2
    else
        nvars=1
        id=1
    end

    # This section removes the drift from a time series using the formula:
    #     drift = (X(T) - X(1)) / (T-1).
    #
    j = collect(1:T);
    Xun=zeros(T,nvars)
    if undrift == true
        if id==2
            for i in 1:nvars
                drift = (X[T,i]-X[1,i])/(T-1);
                Xun[:,i]=X[:,i].-((j.-1)*drift);
            end
        else
            drift = (X[T]-X[1])/(T-1);
            Xun=X.-((j.-1)*drift);
        end
    else
        Xun = X;
    end

    # Create the ideal B's then construct the AA matrix

    b=2*pi/pl;
    a=2*pi/pu;
    bnot = (b-a)/pi;
    bhat = bnot/2;

    B      = (sin.(j*b)-sin.(j*a))./(j*pi);
    B[2:T] = B[1:T-1];
    B[1]   = bnot;

    AA = zeros(2*T,2*T);

    for i=1:T
        AA[i,i:i+T-1] = B';
        AA[i:i+T-1,i] = B;
    end
    AA = AA[1:T,1:T];
    AA[1,1] = bhat;
    AA[T,T] = bhat;

    for i=1:T-1
        AA[i+1,1] = AA[i,1]-B[i,1];
        AA[T-i,T] = AA[i,1]-B[i,1];
    end

    # Filter data using AA matrix and return filtered data
    return(AA*Xun)
end


function hpdi(x, p=5.0;nb_points=20)
# HPDI - Estimates the Bayesian HPD intervals
#

#   Y = HPDI(X,P) returns a Highest Posterior Density (HPD) interval
#   for each column of X. P must be a scalar. Y is a 2 row matrix
#   where ith column is HPDI for ith column of X.

#   References:
#      [1] Chen, M.-H., Shao, Q.-M., and Ibrahim, J. Q., (2000).
#          Monte Carlo Methods in Bayesian Computation. Springer-Verlag.

# Copyright (C) 2001 Aki Vehtari, Adapted to Julia by F. Collard (2019)
#
# This software is distributed under the GNU General Public
# Licence (version 2 or later); please refer to the file
# Licence.txt, included with the software, for details.

    pts=linspace(0.1,99.9-p,nb_points)
    if ndims(x)>1
        m=size(x,2)
        hpdi=zeros(2,m)
        for i in 1:m
            pt1=percentile(x[:,i],pts)
            pt2=percentile(x[:,i],p.+pts)
            cis=abs.(pt2-pt1);
            i_min=argmin(cis);
            hpdi[:,i]=[pt1[i_min],pt2[i_min]]
        end
    else
        m=1
        hpdi=zeros(2)
        pt1=percentile(x,pts)
        pt2=percentile(x,p.+pts)
        cis=abs.(pt2-pt1);
        i_min=argmin(cis);
        hpdi=[pt1[i_min],pt2[i_min]]
    end
    return hpdi
end


function spines_off(ax)
  for sp in ("bottom","top","left","right")
      ax.spines[sp].set_color("none")
  end
end

function fsubplot(r,c,i;bg=(0.95,0.95,0.95),grid=true,fontsize=10,rotation=0,fontname="serif",ylim=[],xlim=[])
  ax=subplot(r,c,i,facecolor=bg)
  if grid
    ax.grid(color="w",linestyle="-",alpha=0.5,linewidth=1.5,zorder=0)
  end
  ax.tick_params(axis="y", width=0, length=0)
  ax.tick_params(axis="x", width=0, length=0)
  setp(ax.get_yticklabels(),family=fontname,size=fontsize)
  setp(ax.get_xticklabels(),family=fontname,size=fontsize,rotation=rotation)
  spines_off(ax)
  # for sp in ("bottom","top","left","right")
  #     ax[:spines][sp][:set_color]("none")
  # end
  if ~isempty(ylim)
    ax.set_ylim(ylim)
  end
  if ~isempty(xlim)
    ax.set_xlim(xlim)
  end
  return ax
end
