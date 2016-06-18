function [ z ] = Tanh( a )
    z=(exp(a)-(exp(-a)))./(exp(a)+(exp(-a)));
end

