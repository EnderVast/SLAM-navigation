%im = imread("img_2.png");

% Get list of all BMP files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.

% Width: 640, Height: 480

%% Superimpose fruit onto arena picture 
mnval = 0.2;
mxval = 0.5;
offset = 30;    % Diagonaly offset fruit from corner of grid
% Left most corner of grids
grid_point = [[0 0]; [213 0]; [426 0]; [0 160]; [213 160]; [426 160]; [0 320]; [213 320]; [426 320]];
% Get left most corner of fruit image
fruit_point = [];

imagefiles = dir('Pictures/Fruits/*.png');
nfiles = length(imagefiles);    % Number of files found

count = 0;
% Save images into images{} and find left most coordinate
for i=1:nfiles
    currentfilename = imagefiles(i).name;
    currentimage = imread("Pictures/Fruits/" + currentfilename);
    images{i} = currentimage;

    % Scale picture first randomly
    scale = mnval + rand*(mxval-mnval);
    current_scaled_image = imresize(images{i}, scale);

    % Find left most corner of fruit
    index = current_scaled_image ~= 0;
    index2 = index(:, :, 1) + index(:, :, 2) + index(:, :, 3);

    max_y = 0;
    min_x = 0;
    for j = 1:480
        row = index2(j, :);
        if (max(row) ~= 0)  % Top most pixel detected
            max_y = j;
            break
        end
    end

    for j = 1:640
        col = index2(:, j);
        if (max(col) ~= 0)  % left most pixel detected
            min_x = j;
            break
        end
    end

    if i == 1
        fruit_point = [max_y min_x];
    else
        fruit_point = [fruit_point; [max_y min_x]];
    end

    current_fruit_point = fruit_point(i, :);
    current_fruit_point_x = fruit_point(i, 2);
    current_fruit_point_y = fruit_point(i, 1);

    % Remove left and top padding
    current_scaled_image = current_scaled_image(current_fruit_point_y - 1: end, current_fruit_point_x - 1: end, :);
    index2 = index2(current_fruit_point_y - 1: end, current_fruit_point_x - 1: end, :);
    % Superimpose
    % Each fruit:
    % For each background, superimpose the fruit on each 9 grid
    % 1. Loop through backgrounds
    % 2. For each background loop, loop through 9 grid points
    % 3. Save picture
    backgroundImage = dir('Pictures/Background/*.png');
    nfiles_bg = length(backgroundImage);    % Number of files found

    for k=1:nfiles_bg
        currentfilename_bg = backgroundImage(k).name;
        currentimage_bg = imread("Pictures/Background/" + currentfilename_bg);
        images_bg{k} = currentimage_bg;

        % Loop through grid
        for j = 1: 9
            start_x_point = grid_point(j, 1) + offset;
            start_y_point = grid_point(j, 2) + offset;

            superimposed_image = currentimage_bg;
            labelled_image = zeros(size(currentimage_bg), 'uint8');

            % Get distance from starting point to edge of picture
            x_dist = 640 - start_x_point;
            y_dist = 480 - start_y_point;

            size_image_x = size(current_scaled_image);
            size_image_x = size_image_x(2);
            size_image_y = size(current_scaled_image);
            size_image_y = size_image_y(1);

            x_diff = x_dist - size_image_x;
            y_diff = y_dist - size_image_y;

            x_bound = start_x_point + size_image_x;
            y_bound = start_y_point + size_image_y;

            if x_diff > 0
                range_x = x_bound;
            else
                range_x = 640;
            end

            if y_diff > 0
                range_y = y_bound;
            else
                range_y = 480;
            end


            x_index = 1;
            y_index = 1;
            for x_bg = start_x_point: range_x - 1
                for y_bg = start_y_point: range_y - 1
                    x_fruit = x_index;
                    y_fruit = y_index;

                    if index2(y_fruit, x_fruit) ~= 0
                        superimposed_image(y_bg, x_bg, :) = current_scaled_image(y_fruit, x_fruit, :);

                        if i <= 12
                            labelled_image(y_bg, x_bg, :) = 5;
                        elseif i >= 13 & i <= 24
                            labelled_image(y_bg, x_bg, :) = 2;
                        elseif i >= 25 & i <= 36
                            labelled_image(y_bg, x_bg, :) = 4;
                        elseif i >= 37 & i <= 48
                            labelled_image(y_bg, x_bg, :) = 3;
                        elseif i >= 49 & i <= 60
                            labelled_image(y_bg, x_bg, :) = 1;
                        end
                    end

                    y_index = y_index + 1;
                end
                y_index = 1;
                x_index = x_index + 1;
            end

            imwrite(labelled_image, "Pictures/labeled/image_" + count + "_label.png");
            imwrite(superimposed_image, "Pictures/Superimposed/image_" + count + ".png");
            count = count + 1;
        end
    end
end
  

%% Red apple
imagefiles = dir('Pictures/RedApple/*.png');      
nfiles = length(imagefiles);    % Number of files found

% Labelling
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread("Pictures/RedApple/" + currentfilename);
   images{ii} = currentimage;
   index = currentimage ~= 0;
   currentimage(index) = 3;
   imwrite(currentimage, "Pictures/labeled/redApple_" + ii + ".png");
end

%% Green apple
imagefiles = dir('Pictures/GreenApple/*.png');      
nfiles = length(imagefiles);    % Number of files found

% Labelling
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread("Pictures/GreenApple/" + currentfilename);
   images{ii} = currentimage;
   index = currentimage ~= 0;
   currentimage(index) = 3;
   imwrite(currentimage, "Pictures/labeled/greenApple_" + ii + ".png");
end

%% Orange
imagefiles = dir('Pictures/Orange/*.png');      
nfiles = length(imagefiles);    % Number of files found

% Labelling
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread("Pictures/Orange/" + currentfilename);
   images{ii} = currentimage;
   index = currentimage ~= 0;
   currentimage(index) = 3;
   imwrite(currentimage, "Pictures/labeled/orange_" + ii + ".png");
end

%% Mango
imagefiles = dir('Pictures/Mango/*.png');      
nfiles = length(imagefiles);    % Number of files found

% Labelling
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread("Pictures/Mango/" + currentfilename);
   images{ii} = currentimage;
   index = currentimage ~= 0;
   currentimage(index) = 3;
   imwrite(currentimage, "Pictures/labeled/mango_" + ii + ".png");
end

%% Capsicum
imagefiles = dir('Pictures/Capsicum/*.png');      
nfiles = length(imagefiles);    % Number of files found

% Labelling
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread("Pictures/Capsicum/" + currentfilename);
   images{ii} = currentimage;
   index = currentimage ~= 0;
   currentimage(index) = 3;
   imwrite(currentimage, "Pictures/labeled/capsicum_" + ii + ".png");
end

