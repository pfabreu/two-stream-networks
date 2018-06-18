function plotColorMatAll(dataSequence , scale)
	n_samples = length(dataSequence.acquisitions);
    fprintf('Number of frames: %d\n', n_samples);
	for n = 1:n_samples
	    dataSample = dataSequence.acquisitions(n);
	    plotColorMat(dataSample, scale);
	end
end    