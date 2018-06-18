root_dir = '/media/pedro/actv3/AHA/Matfiles/Color/'
output_dir = '/media/pedro/actv3/AHA/videos/original/'

files = dir(strcat(root_dir,'*.mat'));
for file = files'
    if isempty(strfind(file.name, 'Body')) && isempty(strfind(file.name, '.avi'))
		matfile = load(strcat(root_dir,file.name));
    		convertVideo(output_dir,file.name,matfile,1.0);
	end
end


function convertVideo(directory, file, dataSequence, scale)
	n_samples = length(dataSequence.acquisitions);
	v = VideoWriter(strcat(directory, file(1:end-4), '.avi'));
	v.open;
	fprintf('Processing video. Number of frames: %d\n', n_samples);
	for n = 1:n_samples
	    dataSample = dataSequence.acquisitions(n).img;
			% Play the video along during conversion
	    %plotColorMat(dataSequence.acquisitions(n), scale);
      v.writeVideo(dataSample);
	end
	v.close;
end



