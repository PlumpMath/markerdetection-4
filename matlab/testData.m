A = imread('inputimg.jpg');
B = rgb2gray(A);



binranges = double(0:255);
C = reshape(B, 1,numel(B));

m = double(255) /double((max(C)-min(C)));
mc = min(C);
D = (C - mc)*m;
bincounts = histc(D,binranges);
D = (B - mc)*m;


close all;
% bar(binranges,bincounts,'histc')

figure;
E = im2bw(D, 0.7);
BI = 1-E;
imshow(1-BI);
hold on;


b = [124 124]';
c = [41 37]';
a = [40 121]';

px = 13;
ac = c-a;
ab = b-a;



dist_px_ab = norm(ab)/px;

p1 = findeEdge( BI, -ac, a);
plot(p1(1), p1(2), 'xg', 'MarkerSize', 20);

p2 = findeEdge( BI, ac, c);
plot(p2(1), p2(2), 'xg', 'MarkerSize', 20);

p3 = findeEdge( BI, -ab, a);
plot(p3(1), p3(2), 'xg', 'MarkerSize', 20);

p4 = findeEdge( BI, ab, b);
plot(p4(1), p4(2), 'xg', 'MarkerSize', 20);


dist_px_ac = norm(p2-p1)/21;
dist_px_ab = norm(p4-p3)/21;

pa_real = p2+ 1*dist_px_ac* (ac/norm(ac)) ...
    - 2.5*dist_px_ab* (ab/norm(ab));
p11 = findeSingleEdge( BI, -ac, pa_real);
p111 = p11 - 2*dist_px_ab* (ab/norm(ab));
plot(p11(1), p11(2), 'xr', 'MarkerSize', 20);
plot(p111(1), p111(2), 'xr', 'MarkerSize', 20);

pa_real = p2- 1*dist_px_ac* (ac/norm(ac)) ...
    - 4.5*dist_px_ab* (ab/norm(ab));
p12 = findeSingleEdge( BI, ab, pa_real);
p121 = p12+ 2 *dist_px_ac* (ac/norm(ac));
plot(p12(1), p12(2), 'xm', 'MarkerSize', 20);
plot(p121(1), p121(2), 'xm', 'MarkerSize', 20);

P1 = p11;
P2 = p111; 
P3 = p12;
P4 = p121;

xs = (P1(1) - P3(1))*(P2(1)*P1(2)-P1(1)*P2(2)) - (P1(1) - P3(1))*(P2(1)*P1(2)-P1(1)*P2(2)) / ...
    


pa_real = p1+ 3.5*dist_px_ac* (ac/norm(ac));
% plot(pa_real(1), pa_real(2), 'xr', 'MarkerSize', 20);

pc_real = p2 - 3.5*dist_px_ac* (ac/norm(ac));
% plot(pc_real(1), pc_real(2), 'xr', 'MarkerSize', 20);

C = p2 - 3.5*dist_px_ab* (ab/norm(ab));
plot(C(1), C(2), 'xr', 'MarkerSize', 20);

pc_real1 = pc_real- 3.5*dist_px_ab* (ab/norm(ab));
% plot(pc_real1(1), pc_real1(2), '*r', 'MarkerSize', 20);

% plot(p3(1), p3(2), 'xr', 'MarkerSize', 20);

A = p1- 3.5*dist_px_ab* (ab/norm(ab));
plot(A(1), A(2), 'om', 'MarkerSize', 20);

pb_real = p4- 3.5*dist_px_ac* (ac/norm(ac));
plot(pb_real(1), pb_real(2), 'or', 'MarkerSize', 20);


B = p4- 3.5*dist_px_ac* (ac/norm(ac));
plot(B(1), B(2), 'og', 'MarkerSize', 20);

v = pc_real1 - p3;

vx = [ B-A];
vx = vx/norm(vx);
vy = [C-A];
vy = vy/norm(vy);
D = (B-A)+C;
plot(D(1), D(2), 'og', 'MarkerSize', 20);

P = [];
for xdist = 0.5:21
    for ydist = 0.5:21
        x = xdist*dist_px_ac*vx;
        y = ydist*dist_px_ab*vy;
        P = horzcat(P, x+y+A);
    end
end

plot(P(1,:), P(2,:), 'xg', 'MarkerSize', 20);


vx = [ B-A; 0];
vx = vx/norm(vx);
vy = [C-A; 0];
vy = vy/norm(vy);
vz = cross(vx, vy);
vz = vz/norm(vz);

vy = cross(vz, vx);
vy = vy/norm(vy);

R = [vx vy vz];






X = X*dist_px_ac;
Y = Y*dist_px_ab;

X = reshape(X, 1, numel(X));
Y = reshape(Y, 1, numel(Y));

V = vertcat(X,Y, zeros(1,numel(X)));
Vs = bsxfun(@plus, R*V,[pa_real1; 0]);


Vs = Vs(1:2,:)';
plot(Vs(:,1), Vs(:,2), 'xg', 'MarkerSize', 20);

