function plotBodyMat( dataSample )
% define the connections, used later when plotting
connections = [1 2; 1 13; 1 17; 2 21; 3 4; 3 21; 5 6; 5 21; 6 7; 7 8; 7 23; 8 22; 9 10; 9 21; 10 11; 11 12; 11 25; 12 24; 13 14; 14 15; 15 16; 17 18; 18 19; 19 20];
colors = ['k','b','r','c','m','g'];
sizes = 30*ones(1,25);
sizes(4)= 200; % head

figure(101);
clf;
% viewpoint specification
az = 0;
el = 90;
view(az, el);
%axis specification
axis([-2,2,-2,2])

n_bodies = length(dataSample.body);
%cycle through bodies
delete(allchild(gca));
hold on;
for i = 1:n_bodies,
    body = dataSample.body(i);
    X = zeros(1,25);
    Y = zeros(1,25);
    Z = zeros(1,25);
    %cycle through joints
    for j=1:25;
        joint = body.joints(j);
        X(j) =  str2double(joint.position(1));
        Y(j) =  str2double(joint.position(2));
        Z(j) =  str2double(joint.position(3));
    end;
    scatter3(X,Y,Z, sizes, colors(mod(i,6)+1));        
    line(X(connections)',Y(connections)',Z(connections)', 'Color', colors(mod(i,6)+1));
end;
hold off;
drawnow;
            