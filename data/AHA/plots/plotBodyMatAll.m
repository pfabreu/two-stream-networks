function plotBodyMatAll( dataSequence )
n_acquisitions = size(dataSequence,1);
for i = 1:n_acquisitions,
n_samples = length(dataSequence(i).acquisitions);
    for n = 1:5:n_samples,
        dataSample = dataSequence(i).acquisitions(n);
        plotBodyMat(dataSample);
    end
end;
           