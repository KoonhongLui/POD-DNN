%   *box* - a vector defining the bounding box of the domain.
%   *hbdy* - the density of the boundary poit-cloud
%   *ptol* - minimum distance between two nodes
%   *ctps* - control-points for density variation 
%   *radius* - distance function for node density metric 


theta = linspace(0, 2*pi, 1000);
r = 0.8 + 0.1 * (sin(6*theta)+sin(3*theta));
x = r.*cos(theta);
y = r.*sin(theta);
cur_xy = zeros(1000, 2);
cur_xy(:, 1) = x;
cur_xy(:, 2) = y;
fileID = fopen('curveshape.txt', 'w');

for i = 1:1000
    fprintf(fileID, '%d\t%d\t%d', cur_xy(i, :));
    fprintf(fileID, '\n');
end

fclose(fileID);

box    = [-1, -1; 1, 1];
hbdy   = 0.05;
ptol   = 0.02;
[b]    = make_domain('curveshape.txt'); 
bdy    = bsmooth(b.xy, hbdy);
ctps   = bdy;
radius = @(p,ctps) 0.035 + 0.05*(min(pdist2(ctps, p)));
[xy]   = NodeLab2D(b.sdf,box,ctps,ptol,radius);
clear b box hbdy ctps radius ptol 
%----------------------------------------------------- 
plot(xy(:,1), xy(:,2),'.k','MarkerSize',12); hold on
plot(bdy(:,1), bdy(:,2), '.k','MarkerSize', 12); axis('square')
set(gca,'visible','off')

save('discretization.mat', 'bdy', 'xy');
% <NodeLab is a simple MATLAB-repository for node-generation and adaptive refinement 
% for testing, and implementing various meshfree methods for solving PDEs in arbitrary domains.>
%     Copyright (C) <2019>  <Pankaj K Mishra>
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
