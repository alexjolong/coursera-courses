function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

cVals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaVals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
numCVals = size(cVals,1);
numSigmaVals = size(sigmaVals,1);
errorMat = zeros(numCVals, numSigmaVals);

for i= 1:numCVals
    for j = 1:numSigmaVals
        model = svmTrain(X, y, cVals(i), @(x1, x2) gaussianKernel(x1, x2, sigmaVals(j)));
        predictions = svmPredict(model, Xval); 
        % predictions is an m x 1 vector where m is the number of training
        % examples in X
        err = mean(double(predictions ~= yval));
        % a double value of the mean error of our prediction from the actual.
        errorMat(i, j) = err;
    end
end

[minVal, elementInd] = min(errorMat(:))
% minVal is our lowest error, and elementInd is the index of the element
% where that error value is found. We extract the row and column from
% errorMat where that error was found, and then get the optimal paramenters
% for C and sigma.
[row, col] = ind2sub(size(errorMat),elementInd)
C = cVals(row)
sigma = sigmaVals(col)


% =========================================================================

end
