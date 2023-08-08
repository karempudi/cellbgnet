function [intCoordsShortAxis, intCoordsLongAxis, cellAreas, cellLengths, cellWidths, cellCounts, intCoordsDepthAxis] = ...
        getDotLocations3D(mCells, varargin)
% This function extracts information about dot locations and is used for
% making fork distribution plots. 
%
% Input:
%   mCells - array of MCell objects.
%   lengthCutOff - cutoff threshold of cell track length. Optional, by
%                  default 5.
%   parentLengthCutoff - cutoff threshold of parent cell track length.
%                        Optional, by default 0.
%   daughterLengthCutoff - cutoff threshold of daughter cell track length.
%                          Optional, by default 0.
%   orientFlag - logical, if 1, reorient cell poles along the trap and
%                recompute internal long axis coordinates. Optional, by
%                default 0.
%   fluoChanName -  String, Optional, If set function returns the dot 
%                   positions only for the particles with 
%                   channelName set to fluoChanName
%
% getDotLocations(...,'PixelSize',pixelSize) scales the output values from
% px to um according to pixelSize (in um). Optional, no scaling by
% default (pixelSize=1).
%
% Output:
%   intCoordsShortAxis - array with dot internal coordinates along 
%                        "short axis"
%   intCoordsLongAxis - array with dot internal coordinates along 
%                        "long axis"
%   cellAreas - array with cell areas in the dot detection frames
%   cellLengths - array with cell lengths in the dot detection frames
%   cellWidths - array with cell widths in the dot detection frames
%   cellCounts - array with cell counts (number of dots in cell in the 
%                correspodning dot detection frame)
%
% See also getDotLocation, displayForkPlot.

ip = inputParser;
validScalarPosNum = @(x) isnumeric(x) && isscalar(x) && (x > 0);
validScalarNonNegNum = @(x) isnumeric(x) && isscalar(x) && (x >= 0);
validLogical = @(x) islogical(x) || (x==1) || (x==0);
validChar = @(x) ischar(x);

ip.addOptional('lengthCutOff',5,validScalarPosNum);
ip.addOptional('parentLengthCutOff',0,validScalarNonNegNum);
ip.addOptional('daughterLengthCutOff',0,validScalarNonNegNum);
ip.addOptional('orientFlag',0,validLogical);
ip.addParameter('PixelSize',1,validScalarPosNum)
ip.addOptional('fluoChanName', '', validChar);

ip.parse(varargin{:});
lengthCutOff = ip.Results.lengthCutOff;
parentLengthCutOff = ip.Results.parentLengthCutOff;
daughterLengthCutOff = ip.Results.daughterLengthCutOff;
orientFlag = ip.Results.orientFlag;
pixelSize = ip.Results.PixelSize;
fluoChanName = ip.Results.fluoChanName;

nCells = length(mCells);
icsas = cell(1,nCells);
iclas = cell(1,nCells);
icdas = cell(1,nCells);
as = cell(1,nCells);
ls = cell(1,nCells);
ws = cell(1,nCells);
ccs = cell(1,nCells);

for i = 1:nCells
    c = mCells(i);
    if ~parentLengthCutOff || ...
            (~isempty(c.parent) && c.parent.lifeTime >= parentLengthCutOff)
        if ~daughterLengthCutOff || ...
                (length(c.descendants) == 2 && all([c.descendants.lifeTime] >= daughterLengthCutOff))
            if c.isBadCell == 0 && c.lifeTime >= lengthCutOff
                [icsas{i}, iclas{i}, icdas{i}, as{i}, ls{i},  ws{i}, ccs{i}] = ...
                    getDotLocation3D(c,'orientFlag', orientFlag, 'fluoChanName', fluoChanName);
            end
        end
    end
end

intCoordsShortAxis = cell2mat(icsas);
intCoordsLongAxis = cell2mat(iclas);
intCoordsDepthAxis = cell2mat(icdas);
cellAreas = cell2mat(as);
cellLengths = cell2mat(ls);
cellWidths = cell2mat(ws);
cellCounts = cell2mat(ccs);

if pixelSize~=1
    cellAreas = cellAreas * pixelSize^2;
    cellLengths = cellLengths * pixelSize;
    cellWidths = cellWidths * pixelSize;
end