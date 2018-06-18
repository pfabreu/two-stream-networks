root_dir = '/media/pedro/actv3/AHA/Matfiles/aha_ist/Color/'
output_dir = '/media/pedro/actv3/AHA/videos/aha_ist/'

files = dir(strcat(root_dir,'*.mat'));
for file = files'
    if isempty(strfind(file.name, 'Body')) && isempty(strfind(file.name, '.avi'))
		matfile = load(strcat(root_dir,file.name));
    		convertVideo(output_dir,file.name,matfile,1.0);
	end
end


function convertVideo(directory, file, dataSequence, scale)
    for i = 1:length(dataSequence.ahaKinectColor)
         seq = dataSequence.ahaKinectColor(i);
         n_samples = length(seq.acquisitions);
         if n_samples ~= 0
            	v = VideoWriter(strcat(directory, file(1:end-4), '.avi'));
            	v.open;
            	fprintf('Processing video. Number of frames: %d\n', n_samples);
            	for n = 1:n_samples
            	    dataSample = seq.acquisitions(n).img;
            			% Play the video along during conversion
            	    plotColorMat(seq.acquisitions(n), scale);
                  v.writeVideo(dataSample);
            	end
            	v.close;
        	end
	end
end



