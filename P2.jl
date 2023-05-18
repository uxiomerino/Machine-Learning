using Statistics
using JLD2
using Images
using Random
using Random:seed!
include("modelComparison.jl")
include("funcionesUtiles.jl")
include("Pupilas_Bordes.jl")


function featureExtraction(ventana::Array{Float64,3})
    caracteristicas = []
    for i in range(1, size(ventana, 3))
        append!(caracteristicas, featureExtraction(ventana[:,:,i]))
    end
    caracteristicas = reshape(caracteristicas, :, 1)
    caracteristicas = convert(Matrix{Real}, caracteristicas)
    return caracteristicas
end

function featureExtraction(ventana::Array{Float64,2})
    caracteristicas = []

    proporcion = size(ventana, 1) / size(ventana, 2)
    push!(caracteristicas, proporcion)

    # media_original = mean(ventana)
    # desv_original = std(ventana)
    # max_original = maximum(ventana)
    # min_original = minimum(filter(!isnan, ventana))
    # push!(caracteristicas, media_original, desv_original, max_original, min_original)

    centro_ventana = centro_imagen(ventana, (0.25, 0.3))
    porcion = round(Int, size(centro_ventana, 2)*0.14)

    centro_left = centro_ventana[:,1:porcion,:]
    centro_middle = centro_ventana[:,porcion:end-porcion,:]
    centro_right = centro_ventana[:,end-porcion:end,:]

    media_left = mean(centro_left)
    desv_left = std(centro_left)
    media_middle = mean(centro_middle)
    desv_middle = std(centro_middle)
    media_right = mean(centro_right)
    desv_right = std(centro_right)

    contraste = media_middle - (media_left + media_right)

    max_centro = maximum(centro_ventana)
    min_centro = minimum(filter(!isnan, centro_ventana))
    # proporcion_centro = size(centro_ventana, 1) / size(centro_ventana, 2)
    push!(caracteristicas, mean(centro_ventana), std(centro_ventana), max_centro, min_centro)

    bordes_ventana = bordes_imagen(ventana, (0.2, 0.1))
    bordes_ventana = filter(!isnan, bordes_ventana)
    media_bordes = mean(bordes_ventana)
    desv_bordes = std(bordes_ventana)
    max_bordes = maximum(bordes_ventana)
    min_bordes = minimum(bordes_ventana)
    # proporcion_bordes = size(bordes_ventana, 1) / size(bordes_ventana, 2)
    push!(caracteristicas, media_bordes, desv_bordes, max_bordes, min_bordes)

    caracteristicas = reshape(caracteristicas, :, 1)
    caracteristicas = convert(Matrix{Real}, caracteristicas)
    return caracteristicas
end

(colorDataset, grayDataset, targets) = loadTrainingDataset()
testDataset = loadTestDataset()
print(typeof(featureExtraction(colorDataset[6])))
print(featureExtraction(colorDataset[6]))

inputs = hcat(featureExtraction.(colorDataset)...);
inputs = convert(Array{Float32,2}, inputs);
inputs = normalizeMinMax(inputs)

seed!(1);

numFolds = 10;

topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento
# Parametros del SVM
kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;
# Parametros del arbol de decision
maxDepth = 4;
# Parapetros de kNN
numNeighbors = 3;

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
modelCrossValidation(:ANN, modelHyperparameters, inputs', targets, numFolds);

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
modelCrossValidation(:SVM, modelHyperparameters, inputs', targets, numFolds);

# Entrenamos los arboles de decision
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs', targets, numFolds);

# Entrenamos los kNN
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs', targets, numFolds);



function modelo_final(ventana::Array{<:Real, 3}, modelo)
    return predict(modelo, featureExtraction(ventana)')
end


# PROBAS

# model = DecisionTreeClassifier(max_depth=maxDepth, random_state=1)
# model = fit!(model, inputs', targets');
# resultados = predict(model, inputs')

model = SVC(kernel="rbf", degree=1, gamma=20, C=1000);
model = fit!(model, inputs', targets');

imagen = testDataset[2]
minWindowSizeY = minimum(size.(colorDataset, 1));
maxWindowSizeY = maximum(size.(colorDataset, 1));
minWindowSizeX = minimum(size.(colorDataset, 2));
maxWindowSizeX = maximum(size.(colorDataset, 2));
windowLocations = Array{Int64,1}[];
for windowWidth = minWindowSizeX:4:maxWindowSizeX
    for windowHeight = minWindowSizeY:4:maxWindowSizeY
        for x1 = 1:10:size(imagen,2)-windowWidth
            for y1 = 1:10:size(imagen,1)-windowHeight
                x2 = x1 + windowWidth;
                y2 = y1 + windowHeight;
                if predict(model, featureExtraction(imagen[y1:y2, x1:x2, :])')[1]
                    push!(windowLocations, [x1, x2, y1, y2]);
                    display(imagen[y1:y2, x1:x2, :])
                end;
            end;
        end;
    end;
end;
