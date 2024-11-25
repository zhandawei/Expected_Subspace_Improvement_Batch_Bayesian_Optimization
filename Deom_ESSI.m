clearvars;clc;close all;
% objective function
fun_name = 'Ellipsoid';
% number of design variables
num_vari = 100;
% lower and upper bounds
lower_bound = -5.12*ones(1,num_vari);
upper_bound = 5.12*ones(1,num_vari);
% number of initial designs
num_initial = 200;
% number of additional function evaluations
addition_evaluation = 1024;
% batch size
batch_size = 64;
max_iteration = round(addition_evaluation/batch_size);
sample_x = lhsdesign(num_initial,num_vari).*(upper_bound-lower_bound)+lower_bound;
sample_y = zeros(size(sample_x,1),1);
for ii = 1:size(sample_x,1)
    sample_y(ii) = feval(fun_name,sample_x(ii,:));
end
evaluation =  size(sample_x,1);
iteration = 1;
[fmin,ind] = min(sample_y);
best_x = sample_x(ind,:);
fprintf('ESSI on %d-D %s, batch size: %d,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,batch_size,iteration-1,evaluation,fmin);
while iteration <= max_iteration
    % train the GP model
    GP_model = GP_train(sample_x,sample_y,lower_bound,upper_bound,1,0.01,100);    
    % select the subspaces randomly
    subspaces = cell(1,batch_size);
    optimized_x = cell(1,batch_size);
    lower = cell(1,batch_size);
    upper = cell(1,batch_size);
    for ii = 1:batch_size
        subspaces{ii} = randperm(num_vari,randi(num_vari));
        lower{ii} = lower_bound(subspaces{ii});
        upper{ii} = upper_bound(subspaces{ii});
    end
    % optimize the ESSI functions
    % you can use parfor here to optimize these ESSI functions in parallel
    infill_x = repmat(best_x,batch_size,1);
    for ii = 1: batch_size
        [optimized_x{ii},max_EI] = Optimizer_GA(@(x)-Infill_ESSI(x,GP_model,fmin,best_x,subspaces{ii}),length(subspaces{ii}),lower{ii},upper{ii},4*length(subspaces{ii}),50);
        infill_x(ii,subspaces{ii}) = optimized_x{ii};
    end
    % evaluate the query points
    % you can use parfor here to evaluate them in parallel
    infill_y = zeros(size(infill_x,1),1);
    for ii = 1:size(infill_x,1)
        infill_y(ii) = feval(fun_name,infill_x(ii,:));
    end
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    iteration = iteration + 1;
    evaluation = evaluation + size(infill_x,1);
    [fmin,ind] = min(sample_y);
    best_x = sample_x(ind,:);
    fprintf('ESSI on %d-D %s, batch size: %d,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,batch_size,iteration-1,evaluation,fmin)
end


