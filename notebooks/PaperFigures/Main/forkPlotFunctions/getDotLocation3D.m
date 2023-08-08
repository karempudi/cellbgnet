function [intCoordsShortAxis, intCoordsLongAxis, intCoordsDepthAxis, dotCellAreas, dotCellLengths, ...
    dotCellWidths, cellCounts, dotCellFrames, cellFluoFrames] = getDotLocation3D(mCell, varargin)
% This function extracts information about dot locations and is used for
% making fork plots. 
%
% Input:
%   mCell - MCell object
%   orientFlag - logical, if 1, reorient cell poles along the trap and
%                recompute internal long axis coordinates
%   fluoIndices - array of phase contrast frames corresponding to 
%                 fluorescence images, optional.
%   fluoChanName -  String, Optional, If set function returns the dot 
%                   positions only for the particles with 
%                   channelName set to fluoChanName
%
% Output:
%   intCoordsShortAxis - array with dot internal coordinates along 
%                        "short axis"
%   intCoordsLongAxis - array with dot internal coordinates along 
%                        "long axis"
%   intCoordsDepthAxis - array with dot internal coordinates along 
%                        "depth axis"
%   dotCellAreas - array with cell areas in the dot detection frames
%   dotCellLengths - array with cell lengths in the dot detection frames
%   dotCellWidths - array with cell widths in the dot detection frames
%   cellCounts - array with cell counts (number of dots in cell in the 
%                correspodning dot detection frame)
%   dotCellFrames - array with dot detection frames, i.e. 
%                   indices of cellFluoFrames, optional. Requires
%                   fluoIndices.
%   cellFluoFrames - array with cell detection frames that have
%                    a corresponding fluo frame. Frame=1 means the cell
%                    birth frame.

ip = inputParser;
validVectorNonNeg = @(x) isnumeric(x) && ismatrix(x) && ~any(x<0);
validLogical = @(x) islogical(x) || (x==1) || (x==0);
validChar = @(x) ischar(x);

ip.addOptional('orientFlag',0, validLogical);
ip.addOptional('fluoIndices', [] ,validVectorNonNeg);
ip.addOptional('fluoChanName', '', validChar) 

ip.parse(varargin{:});
orientFlag = ip.Results.orientFlag;
fluoIndices = ip.Results.fluoIndices;
fluoChanName = ip.Results.fluoChanName;

if isempty(mCell.particles)
    particles = [];
    cellDetectionFrames = [];
else
    if ~isempty(fluoChanName)
        fluoChanNameFcn = str2func(['@(x) strcmp(x, "' fluoChanName '")']);
        selectedParticles = cellfun(fluoChanNameFcn, {mCell.particles.channelName});
        particles = mCell.particles(selectedParticles);
    else
        particles = mCell.particles;
    end
    cellDetectionFrames = [particles.cellDetectionFrames]-mCell.birthFrame+1;
end

cellAreas = mCell.areas(cellDetectionFrames);
cellLengths = mCell.lengths(cellDetectionFrames);
cellWidths = mCell.widths(cellDetectionFrames);
badSegs =  mCell.badSegmentations(cellDetectionFrames);

intCoordsShortAxis = zeros(size(cellDetectionFrames));
intCoordsLongAxis = zeros(size(cellDetectionFrames));
intCoordsDepthAxis = zeros(size(cellDetectionFrames));
dotCellAreas  = zeros(size(cellDetectionFrames));
dotCellLengths = zeros(size(cellDetectionFrames));
dotCellWidths = zeros(size(cellDetectionFrames));
cellCounts = zeros(size(cellDetectionFrames));
dotCellFrames = zeros(size(cellDetectionFrames));

if ~isempty(fluoIndices)  && nargout>=7
    ind = fluoIndices>=mCell.birthFrame & fluoIndices<=mCell.lastFrame;
    cellFluoFrames = fluoIndices(ind)-mCell.birthFrame+1;
end

[~, ~, uidx] = unique(cellDetectionFrames);
uvalCount = accumarray(uidx,1);
frameCounts = uvalCount(uidx);

counter = 0;
for i = 1:length(particles)
    if badSegs(i) == 0 && ~any(isnan(particles(i).trajInternalCoord))
        counter = counter + 1;
        if ~isempty(particles(i).trajInternalCoord)
            cellCounts(counter) = frameCounts(i);
            intCoordsShortAxis(counter) = particles(i).trajInternalCoord(2);
            intCoordsLongAxis(counter) = particles(i).trajInternalCoord(1);
            if size(particles(i).trajImageCoord, 2) == 3
               intCoordsDepthAxis(counter) = particles(i).trajImageCoord(3)/1000; 
            end
        else
            cellCounts(counter) = 1;
            intCoordsShortAxis(counter) = NaN;
            intCoordsLongAxis(counter) = NaN;
            intCoordsDepthAxis(counter) = NaN;
        end
        dotCellAreas(counter) = cellAreas(i);
        dotCellLengths(counter) = cellLengths(i);
        dotCellWidths(counter) = cellWidths(i);
        if ~isempty(fluoIndices)%nargin>=2 && nargout>=7
            dotCellFrames(counter) = find(cellFluoFrames == cellDetectionFrames(i),1);
        end
    end
end

if orientFlag
    counter = 0;
    cellPoles = mCell.poles;
    for i = 1:length(particles)
        if badSegs(i) == 0 && ~any(isnan(particles(i).trajInternalCoord))
            counter = counter + 1;
            curCellPoles = cellPoles{cellDetectionFrames(i)}; 
            if ~isempty(particles(i).trajInternalCoord)
                if curCellPoles(1,1) > curCellPoles(2,1)
                    intCoordsLongAxis(counter) = 1 - intCoordsLongAxis(counter);
                end
            end
        end
    end
end

intCoordsShortAxis = intCoordsShortAxis(1:counter);
intCoordsLongAxis = intCoordsLongAxis(1:counter);
intCoordsDepthAxis = intCoordsDepthAxis(1:counter);
dotCellAreas  = dotCellAreas(1:counter);
dotCellLengths = dotCellLengths(1:counter);
dotCellWidths = dotCellWidths(1:counter);
cellCounts = cellCounts(1:counter);
if ~isempty(fluoIndices)%nargin==3 && nargout>=7
    dotCellFrames = dotCellFrames(1:counter);
end