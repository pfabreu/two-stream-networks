% Go over all videos in directory, get them as a sequence of images
root_dir = '/media/pedro/actv3/AHA/videos/original/'
save_dir = '/media/pedro/actv3/AHA/dynamic_rgb/'
files = dir(strcat(root_dir,'*.avi'));
for file = files'

    vid = VideoReader(strcat(file.folder,'/',file.name));
    frames = read(vid);
    % Get a dynamic image per video
    di = arp(frames);
    % Save the image
    imwrite(di, strcat(save_dir,file.name(1:end-4),'.jpg'))
end

function di = arp(images)
    % Computes approximate dynamic images for a given array of images
    % IMAGES must be a tensor of H x W x D x N dimensionality or
    % cell of image names

    % For the exact dynamic images, use the code
    % http://users.cecs.anu.edu.au/~basura/dynamic_images/code.zip
    % Explained here http://arxiv.org/abs/1512.01848

    if isempty(images)
      di = [] ;
      return ;
    end

    if iscell(images)
      imagesA = cell(1,numel(images)) ;
      for i=1:numel(images)
        if ~ischar(images{i})
          error('images must be an array of images or cell of image names') ;
        end
        imagesA{i} = imread(images{i}) ;
      end
      images = cat(4,imagesA{:}) ;
    end

    N = size(images,4) ;
    di = vl_nnarpooltemporal(single(images),ones(1,N)) ;
end


function Y = vl_nnarpooltemporal(X,ids,dzdy)
    % author: Hakan Bilen
    % approximate rank pooling
    % ids indicates frame-video association (must be in range [1-N])

    sz = size(X);
    forward = logical(nargin<3);

    if numel(ids)~=size(X,4)
      error('Error: ids dimension does not match with X!');
    end

    nVideos = max(ids);

    if forward
      Y = zeros([sz(1:3),nVideos],'like',X);
    else
      Y = zeros(size(X),'like',X);
    end

    for v=1:nVideos
      % pool among frames
      indv = find(ids==v);
      if isempty(indv)
        error('Error: No frames in video %d',v);
      end
      N = numel(indv);
      % magic numbers
      fw = zeros(1,N);
      if N==1
        fw = 1;
      else
        for i=1:N
          fw(i) = sum((2*(i:N)-N-1) ./ (i:N));
        end
      end

      if forward
        Y(:,:,:,v) =  sum(bsxfun(@times,X(:,:,:,indv),...
          reshape(single(fw),[1 1 1 numel(indv)])),4);
      else
        Y(:,:,:,indv) = (bsxfun(@times,repmat(dzdy(:,:,:,v),[1,1,1,numel(indv)]),...
          reshape(fw,[1 1 1 numel(indv)]))) ;
      end
    end
    %
    % if forward
      %   fprintf(' fwd-arpool %.2f ',sqrt(sum(Y(:).^2)));
      % else
      %   fprintf(' back-arpool %f ',sqrt(sum(Y(:).^2)));
    % end
end