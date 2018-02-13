function [outputMat] = hOfX(theta,X)

[xRows,xCols] = size(X); %in our case, X is 5000X400
[thetaRows,thetaCols] = size(theta); %theta is 400X1

% our output will be the same size as X, but every row
% (i.e. every pixel in the training example) is 
% multiplied by theta (i.e. weights for this layer).
outputMat = zeros(xRows,xCols);

% since we are doing element-wise multiplication here,
% we can multiply a 5000X400 matrix with a 1X400 one,
% and have a result which is 5000X400 
Z = X * theta;

% perform sigmoid() on every element of matrix Z
outputMat = sigmoid(z);

end

