function [ pi ] = findeSingleEdge( E, vn, start)
%FINDEEDGE Summary of this function goes here
%   Detailed explanation goes here
vn = vn/norm(vn);

%starts at black
d = 0;
dist = 0.25;
while 1
    p = start+ d*vn;
    pi = floor(p);    
    if E(pi(2), pi(1))
        break;
    end    
    d  = d +dist;
end

 pi = pi-0.5*vn;
end

