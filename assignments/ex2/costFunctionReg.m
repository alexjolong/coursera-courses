function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



function h = hypoth(th, x_to_guess)
    h = sigmoid(th' * x_to_guess);
end

summation = 0;
for i = 1:m
    h_of_xi = hypoth(theta, X(i, :)');
    term1 = -y(i) * log(h_of_xi);
    term2 = (1-y(i)) * log(1 - h_of_xi);
    summation = summation + term1-term2;
    grad = grad + (h_of_xi - y(i))*X(i,:)';
end

reg = 0;
for j=2:n
    reg = reg + theta(j)^2;
end

grad_reg = (lambda / m) * theta;
reg = (reg * lambda) / (2 * m);

grad_reg(1) = 0;

J = (summation / m) + reg;
grad = (grad / m) + grad_reg;




% =============================================================

end
