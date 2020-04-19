% Define State Variables
x1 = SX.sym('x1');
x2 = SX.sym('x2');
x3 = SX.sym('x3');
x4 = SX.sym('x4');
x5 = SX.sym('x5');

states = [x1;x2;x3;x4;x5];
n_states = length(states);

% Define Control Input
u = SX.sym('u');
controls = [u]; 
n_controls = length(controls);

% Define meal disturbance
ra = SX.sym('ra');

% Define system dynamics
rhs = [-(x1*x2);...
        p_2*(s_i*(x3-I_b)-x2);...
       -k_a1*x5);...
       -x5;...
      -(k_a1+k_d)*x5+u]; % system r.h.s

f = Function('f',{states,controls,ra},{rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N); % Decis_ion variables (controls)
P = SX.sym('P',n_states + n_states + N);% parameters (which include the initial and the reference state of the closed loop system)
X = SX.sym('X',n_states,(N+1));% A Matrix that represents the states over the optimization problem.

% compute solution symbolically
X(:,1) = P(1:5); % initial state
for k = 1:N
    st = X(:,k);  
    con = U(:,k);
    D = P(n_states + n_states + k);
    f_value  = f(st,con,D);
    st_next  = st+ (T*f_value);
    X(:,k+1) = st_next;
end
% this function to get the optimal trajectory knowing the optimal solution
ff=Function('ff',{U,P},{X});

obj = 0; % Objective function
g = [];  % constraints vector

Q = zeros(5,5); % weighing matrices (states)
Q(1,1) = 1e10;
Q(2,2) = 1;
Q(3,3) = 1; 
Q(4,4) = 1;
Q(5,5) = 1;
R = zeros(1,1); 
R(1,1) = 1e9; 
% R(2,2) = 0.05; % weighing matrices (controls)

% compute objective
for k=1:N
    st = X(:,k);  
    con = U(:,k);
    obj = obj+(st-P(6:10))'*Q*(st-P(6:10)) + con'*R*con; % calculate obj
end

% compute constraints
for k = 1:N+1   % box constraints due to the map margins
    g = [g ; X(1,k)];   %state x1
end

% make the decision variables one column vector
OPT_variables = reshape(U,1*N,1);
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
% opts.ipopt.max_iter = 100;
% opts.ipopt.print_level =0;%0,3
% opts.print_time = 0;
% opts.ipopt.acceptable_tol =1e-3;
% opts.ipopt.acceptable_obj_change_tol = 1e-3;

solver = nlpsol('solver', 'sqpmethod', nlp_prob,opts);

args = struct;
% inequality constraints (state constraints)
args.lbg = 0;  % lower bound of the states x and y
args.ubg = 190;   % upper bound of the states x and y 

% input constraints
args.lbx(:,1) = u_min; 
args.ubx(:,1) = u_max; 
