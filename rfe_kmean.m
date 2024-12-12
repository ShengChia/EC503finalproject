%% RFE and K-Means Clustering
%Min Tang tming21@bu.edu

%% Load and Normalize Data
filename = 'wdbc.data';
data = readtable(filename, 'FileType', 'text', 'Delimiter', ',');
features = data{:, 3:end};  % Extract features 
labels = data{:, 2};  % Extract diagnosis labels ('M' or 'B')

% Convert labels to binary encoding (1 for 'M', 0 for 'B')
y = double(categorical(labels)) - 1;
X = normalize(features);

featureNames = ["Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness", ...
                "Radius SE", "Texture SE", "Perimeter SE", "Area SE", "Smoothness SE", ...
                "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness", ...
                "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension", ...
                "Compactness SE", "Concavity SE", "Concave Points SE", "Symmetry SE", "Fractal Dimension SE", ...
                "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"];

%% Recursive Feature Elimination (RFE)
numFeatures = size(X, 2);
ranking = zeros(1, numFeatures);  
weights = zeros(1, numFeatures); 
featuresIdx = 1:numFeatures;
thresh = 0.5; 

for i = numFeatures:-1:1
    % Train logistic regression model
    mdl = fitclinear(X(:, featuresIdx), y, 'Learner', 'logistic');
    featureScores = abs(mdl.Beta);
    [~, minIdx] = min(featureScores);
    ranking(featuresIdx(minIdx)) = i;
    weights(featuresIdx(minIdx)) = featureScores(minIdx);
    featuresIdx(minIdx) = [];
end

% Select top features based on threshold
selectedRFEIdx = find(weights >= thresh);
selectedRFEWeights = weights(selectedRFEIdx);
selectedRFEFeatures = featureNames(selectedRFEIdx);

%% K-Means clustering
k = round(sqrt(numFeatures));
clusterCenters = datasample(correlationMatrix, k, 1); % Randomly select k rows from correlation matrix
maxIter = 100;

for iter = 1:maxIter
    % Assign features to nearest cluster
    distances = pdist2(correlationMatrix, clusterCenters); 
    [~, clusterAssignments] = min(distances, [], 2); 

    % Update cluster centers
    newCenters = arrayfun(@(c) mean(correlationMatrix(clusterAssignments == c, :), 1), 1:k, 'UniformOutput', false);
    newCenters = cell2mat(newCenters');

    % Check Convergence 
    if norm(newCenters - clusterCenters) < 1e-6
        break;
    end
    clusterCenters = newCenters;
end

% Select one representative feature per cluster
selectedFeaturesKMeans = zeros(1, k);
selectedFeatureWeightsKMeans = zeros(1, k);
selectedFeatureNamesKMeans = strings(1, k);

for cluster = 1:k
    clusterFeatures = find(clusterAssignments == cluster);
    if ~isempty(clusterFeatures)
        [~, repIdx] = max(mean(abs(correlationMatrix(clusterFeatures, clusterFeatures)), 2));
        selectedFeaturesKMeans(cluster) = clusterFeatures(repIdx);
        selectedFeatureWeightsKMeans(cluster) = mean(abs(correlationMatrix(clusterFeatures, clusterFeatures(repIdx))));
        selectedFeatureNamesKMeans(cluster) = featureNames(clusterFeatures(repIdx));
    else
        warning('Cluster %d is empty; skipping feature selection.', cluster);
    end
end



%% Evaluate K-Means 
cv = cvpartition(y, 'HoldOut', 0.3);
trainIdx = training(cv);
testIdx = test(cv);

X_KMeans = X(:, selectedFeaturesKMeans);
X_KMeans_train = X_KMeans(trainIdx, :);
X_KMeans_test = X_KMeans(testIdx, :);
mdl_KMeans = fitclinear(X_KMeans_train, y(trainIdx), 'Learner', 'logistic');
y_pred_KMeans = predict(mdl_KMeans, X_KMeans_test);
accuracy_KMeans = mean(y_pred_KMeans == y(testIdx));
confMat_KMeans = confusionmat(y(testIdx), y_pred_KMeans);
precision_KMeans = confMat_KMeans(2, 2) / (confMat_KMeans(2, 2) + confMat_KMeans(1, 2));
recall_KMeans = confMat_KMeans(2, 2) / (confMat_KMeans(2, 2) + confMat_KMeans(2, 1));
f1_KMeans = 2 * (precision_KMeans * recall_KMeans) / (precision_KMeans + recall_KMeans);

%% Results
figure(1);
subplot(1, 2, 1);
bar(selectedRFEWeights, 'FaceColor', [0.2 0.6 0.8]);
title('Top Selected Features by RFE');
xlabel('Feature Index');
ylabel('Weight');
xticks(1:length(selectedRFEFeatures));
xticklabels(selectedRFEFeatures);
xlabel('Selected Features');

subplot(1, 2, 2);
bar(selectedFeatureWeightsKMeans, 'FaceColor', [0.8 0.4 0.4]);
title('Selected Features by K-means');
xlabel('Cluster Index');
ylabel('Average Correlation Weight');
xticks(1:k);
xticklabels(selectedFeatureNamesKMeans);

disp('Accuracy and F1-Score Results:');
disp(table(["RFE"; "K-Means"], [accuracies(end); accuracy_KMeans], [f1Scores(end); f1_KMeans], ...
    'VariableNames', {'Method', 'Accuracy', 'F1_Score'}));

