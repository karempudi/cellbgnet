function [mCells]=computeMCellsAndInternalDotCoordinates(coords,segDir,fluoChanName,cellMargin)
% Creates MCell objects for cell masks located in directory segDir and adds
% dot localizations stored in matrix coords.
% Computes internal dot coordinates for future use in forkplots.
%
% Input:
%   coords - dot coordinates matrix with cellbgnet output, columns:
%            frame,x,y,z,photon,prob,err_x,err_y,err_z
%   segDir - directory with sorted cell segmentation images
%   fluoChanName - fluorescence channel name. Default 'fluor'.
%   cellMargin - scalar, cell dilation radius for dot inclusion. Default 2.
%
% Output:
%   mCells - array of MCell objects with detected 3D dots
if size(coords,2)<=8
    error('Matrix coord must have 9 columns')
end
if nargin<3
    fluoChanName='fluor';
end
if nargin<4
    %cellMargin=2;
    cellMargin=0;
end
%% Initiate mCells
import Cell.MCell

distThreshold = 0;

segList = dir(fullfile(segDir,'*.png'));
segList = {segList.name};
nFrames = numel(segList);

mFrames = MCell.mFrames;
mFrames.initialize(nFrames);
cellArrayOfMCells = cell(nFrames,1);
cIdx = 0;
for fIdx = 1:nFrames
    segIm = imread(fullfile(segDir,segList{fIdx}));
    blobLabels = unique(segIm);
    blobLabels = blobLabels(2:end);
    nBlobs = numel(blobLabels);
    if nBlobs>0
        mCells(1:nBlobs,1) = MCell;
        for blobIdx = 1:nBlobs
            cIdx = cIdx+1;
            blobLabel = blobLabels(blobIdx);
            stats = regionprops(segIm == blobLabel,'BoundingBox','Area','Centroid');
            bb = uint16(stats.BoundingBox');
            cropIm = segIm(bb(2):bb(2)+bb(4)-1,bb(1):bb(1)+bb(3)-1);
            blobBoundary = bwboundaries(cropIm == blobLabel,'noholes');
            blobBoundary{1} = blobBoundary{1}(:,[2 1]);
            mCells(blobIdx,1) = Cell.MCell(...
                                    cIdx, fIdx, 1,...
                                    'blobLabels',blobLabel, ...
                                    'boundingBoxes',bb,...
                                    'boundaries', blobBoundary(1),...
                                    'areas', stats.Area, ...
                                    'centroids', stats.Centroid);
            mFrames.addCellPointer(fIdx, mCells(blobIdx), 1);
        end
        cellArrayOfMCells{fIdx} = mCells;
        clear mCells
    else
        mFrames.setPointersInFrame(fIdx,[],{});
    end
end
% Convert cell array of MCells to array of MCells
nCellsPerFrame = cellfun('length',cellArrayOfMCells);
nAllCells = sum(nCellsPerFrame);
mCells(1:nAllCells,1) = MCell;
if nAllCells>0
    start = 1;
    for fIdx = 1:nFrames
        if nCellsPerFrame(fIdx)>0
            finish = nCellsPerFrame(fIdx)+start-1;
            mCells(start:finish) = cellArrayOfMCells{fIdx};
            start = finish+1;
        end
    end
end
disp('Initiated m cell objects ...');
%% Compute cell measurements
import Cell.Measure.*


dilationRadius = floor(distThreshold/2);
se = strel('disk', dilationRadius, 8);

for fIdx=1:nFrames
    currSegImage = imread(fullfile(segDir,segList{fIdx}));
    blobLabels = mFrames.getBlobsInFrame(fIdx);
    for bIdx = 1:numel(blobLabels)
        blobLabel = blobLabels(bIdx);
        
        c = mFrames.getCellsInBlob(fIdx, blobLabel);
        t = c.getIndex(fIdx);

        % Crop the cell image from the frame in a bounding box
        cellIm = cropCell(currSegImage,c,t,c.isRotated(t));

        if ~c.badSegmentations(t)
            % Compute cell length by fitting a curve into cell backbone
            backbone = computeCellBackbone(cellIm, distThreshold, se);
            if backbone.length > 0
                c.lengths(t) = backbone.length;
                c.widths(t) = computeCellWidth(cellIm, backbone);
                c.poles{t} = c.convertFromCellCoord(t, backbone.poles, 1);
                c.backbones(:,t) = backbone.fitCoeff;
            end
        else
            c.lengths(t) = -1;
        end
    end
end
disp('Computed cell measurements');
%% Add dot localizations to mCells
rangeFluoStack = 1;
if cellMargin > 0
    marginStrel = strel('disk',cellMargin);
end

% Convert to decode coords: [x y frame z errx erry errz bg ampl]
% 'image','x','y','z','photon','prob','sigma_x','sigma_y','sigma_z','sigma_photon'
totalDotCoords = cell(nFrames,1);
for fluoIdx=1:nFrames
    dots=coords(coords(:,1)==fluoIdx,:);
    N=size(dots,1);
    dots=[dots(:,[2 3]) ones(N,1) dots(:,[4 7:9]) nan(N,1) dots(:,5) dots(:, 6)];
    totalDotCoords{fluoIdx}=dots;
end

mFrames=Cell.MCell.mFrames;
T=eye(3); % one to one correspondance between fluo and phase channels
for fluoIdx = 1:nFrames
    phaseFrame=fluoIdx;
    dotsInFluo = totalDotCoords{fluoIdx};
    dotsInPhase = convertCoord(dotsInFluo,T);
    blobsInRoi = mFrames.getBlobsInFrame(fluoIdx);
    
    % discard dots outside cells
    currSegIm = imread(fullfile(segDir,segList{fluoIdx}));
    allCellsIm = ismember(currSegIm,uint16(blobsInRoi));
    if cellMargin>0
        allCellsIm = imdilate(allCellsIm,marginStrel);
    end
    dotsInPhase = BuildTrajectories.findPointsInCell(dotsInPhase(:,1:3),rangeFluoStack,allCellsIm,1,[],dotsInPhase(:,4:end));
    %dotsInFluo = convertCoord(dotsInPhase,T,1);
    
    % Determine cell identity of all spots, and discard spots outside cells
    nBlobs = length(blobsInRoi);
    dotTrajs = cell(nBlobs,1);
    dotData = [];
    for blobIdx = 1:nBlobs
        % Crop and dilate cell image
        currBlob = blobsInRoi(blobIdx);
        cellsInBlob = mFrames.getCellsInBlob(phaseFrame, currBlob);
        c = cellsInBlob(1);
        t = c.getIndex(phaseFrame);
        cellBoundingBox = double(c.boundingBoxes(:,t));
        cellIm = Cell.Measure.cropCell(currSegIm,c,t);
        if cellMargin>0
            cellIm = padarray(cellIm, [cellMargin cellMargin]);
            cellIm = imdilate(cellIm,marginStrel);
        end
        offset = cellBoundingBox(1:2)-cellMargin-1;
        % Select dots in the cell
        cellDotsInPhase = BuildTrajectories.findPointsInCell(...
            dotsInPhase(:,1:3),rangeFluoStack,cellIm,currBlob,offset,dotsInPhase(:,5:end));
        if ~isempty(cellDotsInPhase)
            cellDotsInFluo = convertCoord(cellDotsInPhase,T,1);
            % Discard movie frames with number of dots in cell above threshold
%             cellDotsInFluo = Spot.Refine.filterDotsInCell(cellDotsInFluo,maxFluoFrame,maxDotsInCell);
            cellDotsInFluo(:,5) = []; % remove tmp id column
            dotData = [dotData; cellDotsInFluo];
            % Build dot trajectories
            dotTrajs{blobIdx} = num2cell(cellDotsInFluo,2);
        end
    end
%    dotTrajs = Spot.Refine.curateDuplicatedDots(dotTrajs,blobsInRoi,dotData,rangeFluoStack,parameters);
 
    % Update MCell objects
    for blobIdx = 1:nBlobs
        currBlob = blobsInRoi(blobIdx);  
        cellsInBlob = mFrames.getCellsInBlob(phaseFrame, currBlob);
        c = cellsInBlob(1);
        if isempty(dotTrajs)
            dotTrajsInCell = [];
        else
            dotTrajsInCell = dotTrajs{blobIdx};
        end
        if ~isempty(dotTrajsInCell)
            nTrajs = numel(dotTrajsInCell);
            for trajIdx = 1:nTrajs
                currTraj = dotTrajsInCell{trajIdx};
                trajLength = size(currTraj,1);
                pIdx = length(c.particles) + 1;
                c.particles(pIdx).channelName = fluoChanName;
                c.particles(pIdx).firstFrame = currTraj(1,3);
                c.particles(pIdx).id = currTraj(:,5);
                c.particles(pIdx).trajImageCoord = currTraj(:,[1:2 6]);
                c.particles(pIdx).err = currTraj(:,7:9);
                c.particles(pIdx).prob = currTraj(:, 12);
                c.particles(pIdx).bg = currTraj(:,10);
                c.particles(pIdx).ampl = currTraj(:,11);
                c.particles(pIdx).trajPhaseImageCoord = convertCoord(c.particles(pIdx).trajImageCoord,T);
                c.particles(pIdx).cellDetectionFrames = repmat(phaseFrame,trajLength,1);
                c.particles(pIdx).blobLabels = repmat(currBlob,trajLength,1);
                c.particles(pIdx).numberOfFrame = trajLength;
                c.particles(pIdx).width = c.widths(1);
                c.particles(pIdx).length = c.lengths(1);
            end
        else
             % Mark that cell is inside fluo roi but has no dots
            pIdx = length(c.particles)+1;
            c.particles(pIdx).channelName = fluoChanName;
            c.particles(pIdx).trajImageCoord = [];
            c.particles(pIdx).cellDetectionFrames = phaseFrame;
            c.particles(pIdx).blobLabels = currBlob;
            c.particles(pIdx).numberOfFrame = 0;
            c.particles(pIdx).prob = -1;
            c.particles(pIdx).width = -1;
            c.particles(pIdx).length = -1;
        end
    end
end
disp('Converted to decode coordinates ...');
%% Final step: Compute internal dot coordinates
Cell.Measure.computeParticleInternalCoord(mCells);