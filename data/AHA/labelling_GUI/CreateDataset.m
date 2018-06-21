function varargout = CreateDataset(varargin)
% CREATEDATASET MATLAB code for CreateDataset.fig
%      CREATEDATASET, by itself, creates a new CREATEDATASET or raises the existing
%      singleton*.
%
%      H = CREATEDATASET returns the handle to a new CREATEDATASET or the handle to
%      the existing singleton*.
%
%      CREATEDATASET('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CREATEDATASET.M with the given input arguments.
%
%      CREATEDATASET('Property','Value',...) creates a new CREATEDATASET or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CreateDataset_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CreateDataset_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CreateDataset

% Last Modified by GUIDE v2.5 14-Mar-2018 18:17:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CreateDataset_OpeningFcn, ...
                   'gui_OutputFcn',  @CreateDataset_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before CreateDataset is made visible.
function CreateDataset_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to CreateDataset (see VARARGIN)

% Choose default command line output for CreateDataset
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes CreateDataset wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = CreateDataset_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in openFileButton.
function openFileButton_Callback(hObject, eventdata, handles)
% hObject    handle to openFileButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

handles.skeletonPath = '/media/pedro/actv4/AHA/AHA2/Skeleton';
handles.rgbPath = '/media/pedro/actv4/AHA/AHA2/RGB';
handles.skeletonFilename = uigetfile(handles.skeletonPath);
handles.skeletonSequence = load([handles.skeletonPath '/' handles.skeletonFilename]);
handles.rgbFilename = [handles.skeletonFilename(1:end-8) 'Color.mat'];
handles.rgbSequence = load([handles.rgbPath '/' handles.rgbFilename]);
rgbTimes = zeros(length(handles.rgbSequence.acquisitions),1);
for i = 1 : length(handles.rgbSequence.acquisitions)
    rgbTimes(i) = handles.rgbSequence.acquisitions(i).time;
end
handles.rgbTimes = rgbTimes';
set(handles.staticText1,'String', ['Now Labelling: ' handles.skeletonFilename])
set(handles.TextRGB,'String', ['Now Labelling: ' handles.rgbFilename])

handles.labels = zeros(1,length(handles.skeletonSequence.acquisitions));
handles.currentFrame = 1;
handles.currentLabel = 1;
handles.lastFrameNumber = length(handles.skeletonSequence.acquisitions);

set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
plotBodyMat(currentSkeleton,handles)
[~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
plotColorMat(currentRGB,1,handles)

set(handles.previousFrameButton, 'Visible', 'On');
set(handles.nextFrameButton, 'Visible', 'On');
set(handles.forward30FramesButton, 'Visible', 'On');
set(handles.back15FramesButton, 'Visible', 'On');
set(handles.saveButton, 'Visible', 'On');
set(handles.goToFrameButton, 'Visible', 'On');
set(handles.showLabelsButton,'Visible', 'On');
set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
guidata(hObject,handles)

% --- Executes on button press in previousFrameButton.
function previousFrameButton_Callback(hObject, eventdata, handles)
% hObject    handle to previousFrameButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.currentFrame == 1
    disp('First frame Reached');
else
    handles.currentFrame = handles.currentFrame-1;
    set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
    set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
    set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
    currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
    plotBodyMat(currentSkeleton,handles)
    [~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
    currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
    plotColorMat(currentRGB,1,handles)
    handles.labels(handles.currentFrame+1) = handles.currentLabel;
end
updateRecentLabelsWheel(handles);
guidata(hObject,handles)


% --- Executes on button press in nextFrameButton.
function nextFrameButton_Callback(hObject, eventdata, handles)
% hObject    handle to nextFrameButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.currentFrame == handles.lastFrameNumber
    disp('Last frame Reached');
else
    handles.currentFrame = handles.currentFrame+1;
    set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
    set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
    set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
    currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
    plotBodyMat(currentSkeleton,handles)
    [~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
    currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
    plotColorMat(currentRGB,1,handles)
    handles.labels(handles.currentFrame-1) = handles.currentLabel;
end
updateRecentLabelsWheel(handles);
guidata(hObject,handles)

% --- Executes on button press in back15FramesButton.
function back15FramesButton_Callback(hObject, eventdata, handles)
% hObject    handle to back15FramesButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.currentFrame-15 <= 1
    disp('First frame Reached');
else
    handles.currentFrame = handles.currentFrame-15;
    set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
    set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
    set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
    currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
    plotBodyMat(currentSkeleton,handles)
    [~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
    currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
    plotColorMat(currentRGB,1,handles)
    handles.labels(handles.currentFrame +1 : handles.currentFrame + 15) = handles.currentLabel;
end
updateRecentLabelsWheel(handles);
guidata(hObject,handles)

% --- Executes on button press in forward30FramesButton.
function forward30FramesButton_Callback(hObject, eventdata, handles)
% hObject    handle to forward30FramesButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if handles.currentFrame+30 >= handles.lastFrameNumber
    disp('Last frame Reached');
    oldFrame = handles.currentFrame;
    handles.currentFrame = handles.lastFrameNumber;
    set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
    set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
    set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
    currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
    plotBodyMat(currentSkeleton,handles)
    [~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
    currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
    plotColorMat(currentRGB,1,handles)
    handles.labels(oldFrame : handles.currentFrame) = handles.currentLabel;
else
    handles.currentFrame = handles.currentFrame+30;
    set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
    set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
    set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
    currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
    plotBodyMat(currentSkeleton,handles)
    [~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
    currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
    plotColorMat(currentRGB,1,handles)
    handles.labels(handles.currentFrame - 30 : handles.currentFrame - 1) = handles.currentLabel;
end
updateRecentLabelsWheel(handles);
guidata(hObject,handles)

% --- Plots RGB Image for the current Frame.
function plotColorMat( dataSample , scale,handles )
axes(handles.axes2)
cla;
imshow(imresize(dataSample.img,scale));
drawnow;

%--- Plots Skeleton for the current Frame
function plotBodyMat( dataSample ,handles)
% define the connections, used later when plotting
axes(handles.axes1)
connections = [1 2; 1 13; 1 17; 2 21; 3 4; 3 21; 5 6; 5 21; 6 7; 7 8; 7 23; 8 22; 9 10; 9 21; 10 11; 11 12; 11 25; 12 24; 13 14; 14 15; 15 16; 17 18; 18 19; 19 20];
colors = ['k','b','r','c','m','g'];
sizes = 30*ones(1,25);
sizes(4)= 200; % head

%clf;
% viewpoint specification
az = 0;
el = 90;
view(az, el);
%axis specification
axis([-2,2,-2,2])

n_bodies = length(dataSample.body);
%cycle through bodies
%delete(allchild(gca));
hold on;
for i = 1:n_bodies,
    body = dataSample.body(i);
    X = zeros(1,25);
    Y = zeros(1,25);
    Z = zeros(1,25);
    %cycle through joints
    for j=1:25;
        joint = body.joints(j);
        X(j) =  str2double(joint.position(1));
        Y(j) =  str2double(joint.position(2));
        Z(j) =  str2double(joint.position(3));
    end;
    cla;
    scatter3(handles.axes1,X,Y,Z, sizes, colors(mod(i,6)+1));        
    line(X(connections)',Y(connections)',Z(connections)', 'Color', colors(mod(i,6)+1));
end;
hold off;
drawnow;


function updateRecentLabelsWheel(handles)
if handles.currentFrame > 4 && handles.currentFrame < handles.lastFrameNumber - 4
    set(handles.previousLabelsText,'Visible','On');
    set(handles.currentLabelText,'Visible','On');
    set(handles.nextLabelsText,'Visible','On');

   set(handles.previousLabelsText,'String', [num2str(handles.labels(handles.currentFrame-3)) ' ' num2str(handles.labels(handles.currentFrame-2)) ' ' num2str(handles.labels(handles.currentFrame-1))]);
   set(handles.currentLabelText,'String', num2str(handles.labels(handles.currentFrame)));
   set(handles.nextLabelsText,'String', [num2str(handles.labels(handles.currentFrame+1)) ' ' num2str(handles.labels(handles.currentFrame+2)) ' ' num2str(handles.labels(handles.currentFrame+3))]);
else
    set(handles.previousLabelsText,'Visible','Off');
    set(handles.currentLabelText,'Visible','Off');
    set(handles.nextLabelsText,'Visible','Off');
end
    



% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
handles.currentLabel = get(hObject, 'Value');
set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
guidata(hObject,handles)


% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in saveButton.
function saveButton_Callback(hObject, eventdata, handles)
% hObject    handle to saveButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
labels = handles.labels';
for i =1:length(labels)
    if (labels(i) == 0 )
        labels(i) = 1;
    end
end
savePath = '/media/pedro/actv4/AHA/AHA2/Labels/labels-pedro/';
handles.rgbFilename = [handles.skeletonFilename(1:end-8) 'Color.mat'];

saveFilename = [savePath handles.skeletonFilename(1:end-8) 'Labels.mat'];
save(saveFilename,'-v7.3','labels');



% --- If Enable == 'on', executes on mouse press in 5 pixel border.
% --- Otherwise, executes on mouse press in 5 pixel border or over openFileButton.
function openFileButton_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to openFileButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in goToFrameButton.
function goToFrameButton_Callback(hObject, eventdata, handles)
% hObject    handle to goToFrameButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
answer = -1;
while(answer < 1 || answer > handles.lastFrameNumber)
    prompt = {['Enter the frame you wish to go to (between: 1 and ' num2str(handles.lastFrameNumber) ')']};
    dlg_title = 'Go to Frame';
    num_lines = 1;
    defaultans = {'1'};
    answer = inputdlg(prompt,dlg_title,num_lines,defaultans);
    answer = str2num(answer{1});
    if answer > handles.lastFrameNumber || answer < 1
        errordlg(['Please choose a frame number between 1 and ' num2str(handles.lastFrameNumber) '!'],'Stop being silly!')
    end
end
handles.currentFrame = answer;
set(handles.oldLabelText,'String', ['Old Label:' num2str(handles.labels(handles.currentFrame))])
set(handles.newLabelText,'String',['New Label:' num2str(handles.currentLabel)]);
set(handles.frameNumberText,'String',['Frame #' num2str(handles.currentFrame) ' of ' num2str(handles.lastFrameNumber)]);
currentSkeleton = handles.skeletonSequence.acquisitions(handles.currentFrame);
plotBodyMat(currentSkeleton,handles)
[~,rgbFrame] = ismember(currentSkeleton.time,handles.rgbTimes);
currentRGB = handles.rgbSequence.acquisitions(rgbFrame);
plotColorMat(currentRGB,1,handles)
guidata(hObject,handles)
updateRecentLabelsWheel(handles)


% --- Executes on button press in showLabelsButton.
function showLabelsButton_Callback(hObject, eventdata, handles)
% hObject    handle to showLabelsButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1)
cla;
x = 1:length(handles.labels);
plot(x,handles.labels);
