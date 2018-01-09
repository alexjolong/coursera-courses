function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% since this is simple linear regression, our hypothesis is just theta' * x
% size(theta) is 2x1
% size(X) is 12x2
% size(y) is 12x1
h = bsxfun(@times,X,theta'); % this may be a little faster than a loop
h = sum(h,2); % returns a column vector of the sums of each row
J = (1/(2*m)) * sum((h - y).^2);

regTheta = theta;
regTheta(1) = 0;
reg = (lambda / (2 * m)) * sum(regTheta.^2);

J = J + reg;

% =========================================================================

grad = grad(:);

end
