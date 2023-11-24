clear;

%% Inputs with H0

rng(1023)

nd = 5;
h = [0, 1, 2, 3, 6];

name = 'hutils_with_h0';

generate_test(nd, h, name)

%% Inputs without H0 

rng(32767)

nd = 7;
h = [1, 2, 3, 5, 7, 9];

name = 'hutils_without_h0';

generate_test(nd, h, name)

%% Generator Function

function [] = generate_test(nd, h, name)

    M = rand(nd, nd);
    C = rand(nd, nd);
    K = rand(nd, nd);

    w = rand(1);

    Nt = 2^7;

    Nhc = 2*sum(h ~= 0) + sum(h==0);
    X0 = rand(Nhc, nd);

    [E,dEdw] = HARMONICSTIFFNESS(M,C,K,w,h);

    ord = 0;
    x_t0 = TIMESERIES_DERIV(Nt,h,X0,ord);
    v0 = GETFOURIERCOEFF(h, x_t0);

    ord = 1;
    x_t1 = TIMESERIES_DERIV(Nt,h,X0,ord);
    v1 = GETFOURIERCOEFF(h, x_t1);

    ord = 2;
    x_t2 = TIMESERIES_DERIV(Nt,h,X0,ord);
    v2 = GETFOURIERCOEFF(h, x_t2);

    ord = 3;
    x_t3 = TIMESERIES_DERIV(Nt,h,X0,ord);
    v3 = GETFOURIERCOEFF(h, x_t3);

    clear ord Nhc

    save(name)
end


