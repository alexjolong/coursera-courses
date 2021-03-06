function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    numFeatures = size(X,2);
    sum = zeros(numFeatures,1);
    for i=1:m
        hOfX = theta' * X(i,:)'; %hOfX is a scalar (predicted value)
        diff = hOfX - y(i); %diff is a scalar (error)
        sum = sum + (diff * X(i,:)'); %rolling sum of error * features of set i
    end
    
    partialDeriv = (1/m) * sum;
    theta = theta - (alpha * partialDeriv);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
