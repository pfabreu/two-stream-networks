load('pred.mat')
for i = 1:length(xmldata)
	xmldata{i}.boxes = cell2mat(xmldata{i}.boxes);
	xmldata{i}.framenr = cell2mat(xmldata{i}.framenr);
end
xmldata = cell2mat(xmldata);