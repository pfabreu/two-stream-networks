actions = {'Basketball','BasketballDunk','Biking','CliffDiving','CricketBowling',...
        'Diving','Fencing','FloorGymnastics','GolfSwing','HorseRiding','IceDancing',...
        'LongJump','PoleVault','RopeClimbing','SalsaSpin','SkateBoarding','Skiing',...
        'Skijet','SoccerJuggling','Surfing','TennisSwing','TrampolineJumping',...
        'VolleyballSpiking','WalkingWithDog'};

preds = {'../pred_crop.mat','../pred_fovea.mat','../pred_gauss.mat','../pred_RGB.mat','../pred_flow.mat',...
		'../predfusion_RGB.mat', '../predfusion_crop.mat', '../predfusion_fovea.mat', '../predfusion_gauss.mat'};

for j = 1:length(preds)
	load(preds{j})
	for i = 1:length(xmldata)
		xmldata{i}.boxes = cell2mat(xmldata{i}.boxes);
		xmldata{i}.framenr = cell2mat(xmldata{i}.framenr);
	end
	xmldata = cell2mat(xmldata);

	%results = load('saha-bmvc-detections.mat');
	%detections = results.xmldata;
	detections = xmldata;
	load('../../../data/UCF101-24/testlist.mat')

	%% LOAD bmvc annot and eval above detections
	%annot = load('../annot_bmvc.mat');
	%mAPs = zeros(200,6); new = 0; count = 0;
	% load('temp.mat');
	%for iou_th = [0.2,0.5:0.05:0.95]
	%    [smAP,smIoU,sacc,sAP] = get_PR_curve(annot.videos, sahadetection, testlist, actions,  iou_th);
	%    [ffmAP,ffmIoU,ffacc,ffAP] = get_PR_curve(annot.videos, fastflowDetections, testlist, actions,  iou_th);
	%    [sfmAP,sfmIoU,sfacc,sfAP] = get_PR_curve(annot.videos, slowflowDetections, testlist, actions,  iou_th);
	%    [pmAP,pmIoU,pacc,pAP] = get_PR_curve(annot.videos, pengdetection, testlist, actions,  iou_th);
	%    count = count + 1;
	%    mAPs(count,:) = [new,iou_th, smAP, ffmAP, sfmAP,pmAP];
	%    fprintf('IOU-TH %.2f SAHA %0.3f FASTFLOW %0.3f SLOWFLOW %0.3f PENG %0.3fN\n',iou_th,smAP,ffmAP,sfmAP,pmAP);
	%end 
	% save('temp.mat','mAPs','count');
	%% LOAD new annots and eval above detections
	annot = load('../../../data/UCF101-24/finalAnnots.mat');
	iou_th = 0.5;
	[mAP,mIoU,acc,AP] = get_PR_curve(annot.annot, detections, testlist, actions,  iou_th);
	fprintf('%s: IOU-TH %.2f mAP %0.3f \n',preds{j},iou_th,mAP);
end
%save('mAPs.mat','mAPs');
