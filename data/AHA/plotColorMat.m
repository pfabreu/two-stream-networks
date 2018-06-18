function plotColorMat( dataSample , scale )
	figure(104);
	imshow(imresize(dataSample.img,scale));
	drawnow;
end