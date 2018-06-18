skList = dir('Skeleton');
skList = skList(4:end);
skPath = 'Skeleton/';
for i = 13:length(skList)
    data = load([skPath skList(i).name]);
    skeletons = zeros(length(data.acquisitions),25*3);
    for frame = 1:length(data.acquisitions)
        if isempty(data.acquisitions(frame).body)
            disp('Found an empty Skeleton');
        else
            joints = data.acquisitions(frame).body.joints;
            for joint = 1 : length(joints)
                currentJoint = [str2double(joints(joint).position(1)) str2double(joints(joint).position(2)) str2double(joints(joint).position(3))];
                skeletons(frame,3*(joint-1)+1:3*(joint-1)+3) = currentJoint;
            end
        end
    end
    saveFilename = ['Skeletons_Python/' skList(i).name];
    save(saveFilename,'-v7.3','skeletons');
end