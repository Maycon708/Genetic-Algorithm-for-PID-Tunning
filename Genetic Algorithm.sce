numBits = 10;           // numero de bits para cada parametro
minBound = 0;           // menor valor que um parametro pode assumir
maxBound = 20;          // maior valor que um parametro pode assumir
popSize = 10;           // tamanho da população
nGenerations = 100;     // numero de gerações
nSamples = 100;         // numero de amostras para determinação do fitness
tSamples = 0.1;         // tempo entre as amostras
mutateRate = 1;         // probabilidade de mutação
crossoverRate = 0.8;    // probabilidade de cruzamento

/*
*   Adiciona um parametro ao individuo
*   - Se valor < 0 adiciona um valor aleatorio entre 0 e 2^numBits
*/
function tmp = newParam(individual, param, valor)
    tmp = individual;
    if valor < 0 then
        valor = grand('uin', 0, 2^numBits);
    end
    tmp = add_param(tmp, param, valor);
endfunction

/*
*   Gera um individuo com base nos valores kp, ki e kd
*   - Para valores menores que zero,
*       gera com um valor aleatorio entre 0 e 2^numBits
*/
function individual = generateIndividual(kp, ki, kd)
    individual = init_param();
    individual = newParam(individual, 'kp', kp);
    individual = newParam(individual, 'ki', ki);
    individual = newParam(individual, 'kd', kd);
endfunction

/*
*   Gera uma lista contendo a população inicial
*/
function popInitial = generatePopInitial(popSize)
    popInitial = list();
    for i = 1:popSize
        popInitial(i) = generateIndividual(-1, -1, -1);
    end
endfunction

/*
*   Recebe um individuo contendo Kp, Ki e Kd
*       e retorna a equação PID no dominio S
*       correspondente ao mesmo
*/
function PID = transformToPID(individual)
    kp = get_param(individual, 'kp')/1024*(maxBound-minBound)+minBound;
    ki = get_param(individual, 'ki')/1024*(maxBound-minBound)+minBound;
    kd = get_param(individual, 'kd')/1024*(maxBound-minBound)+minBound;
    s = poly(0, 's');
    PID = syslin('c', (kp*s+ki+kd*s^2)/(s));
endfunction

/*
*   Calcula o quão apto um individuo esta para resolver o problema
*   - Obtem-se a equação PID do individuo no dominio S
*   - Gera-se equação do ganho de realimentação
*   - Gera-se a euqação de malha fechada
*       -> HS * PID com realimentação unitaria
*   - Obtem-se a resposta de GS a um degrau unitario
*   - A aptidão de um individuo é a inversa da integral de |erro|*tempo
*       definida de 0 até tempo total da simulação,
*       sendo erro a diferença entre o valor atual e o valor esperado
*/
function fitness = getFitness(individual)
    PID = transformToPID(individual);
    s = poly(0, 's');
    feedback = syslin('c', s/s);
    GS = (HS*PID)/.feedback;
    t = 0:tSamples:nSamples*tSamples
    answer = csim('step', t, GS);
    fitness = 0;
    for i = 1:nSamples
        erro = 1-answer(i);
        fitness = fitness + (abs(erro) * t(i));
    end
    fitness = 1/fitness;
endfunction

/*
* Gera uma lista contendo a aptidão de cada individuo
*/
function fitnessList = getAllFitness(population)
    fitnessList = list();
    for i = 1:popSize
        fitnessList(i) = getFitness(population(i));
    end     
endfunction

/*
*   Retorna uma lista com a probabilidade de cada individuo
*       ser escolhido durante o processo de seleção
*   - Tal probabilidade é diretamente proporcional a 
*       aptidao do individuo
*/
function normalizedList = normalize(fitnessList)
    normalizedList = list();
    sumFitness = 0;
    for i = 1:popSize
        sumFitness = sumFitness + fitnessList(i);
    end
    for i = 1:popSize
        normalizedList(i) = fitnessList(i)/sumFitness;
    end
endfunction

/*
*   Retorna o melhor individuo, ou seja, com a maior aptidão 
*/
function best = getBest(population)
    fitnessList = getAllFitness(population);
    bestId = 1;
    for i = 1:popSize
        if fitnessList(i) > fitnessList(bestId) then
            bestId = i;
        end
    end
    best = population(bestId);
endfunction

/*
*   Seleciona os individuos para o cruzamento
*   - Baseado no processo de seleção por roleta
*   - A probabilidade de um individuo ser selecionada é
*       diretamente proporcional ao quão apto ele está
*/
function parents = selection(population, probability)
    parents = init_param();
    probabilityA = grand('def');
    probabilityB = grand('def');
    sumProbability = 0;
    for i = 1:popSize
        sumProbability = sumProbability + probability(i);
        if sumProbability >= probabilityA then
            parentA = population(i);
        end
        if sumProbability >= probabilityB then
            parentB = population(i);
        end
    end
    parents = add_param(parents, 'A', parentA);
    parents = add_param(parents, 'B', parentB);
endfunction

/*
*   Função auxiliar que retorna a mascara de bits 
*       a ser utilizada no cruzamento
*/
function mask = getMask(ini, fim)
    mask = 0;
    for i = ini:fim
        mask = bitset(mask, i);
    end
endfunction

/*
*   Retorna o cruzamento de um determinado parâmetro
*   - Determina de forma aleatória o quanto o filho vai 
*       herdar de cada pai
*   - Tal valor é medido pela quantidade de bits do pai que serão
*       transmitidos ao filho
*   - O filho sempre herda ao menos um bit de cada pai
*/
function son = getCross(parentA, parentB)
    pos = grand('uin', 1, numBits);
    maskA = getMask(1, pos);
    maskB = getMask(pos+1, numBits);
    son = bitor(bitand(parentA, maskA), bitand(parentB, maskB));
endfunction

/*
*   Dados os pais escolhidos pela seleção, a fução determina,
*       o resultado do cruzamento, se o mesmo acontecer
*   - Caso não haja cruzamento o filho é uma copia exata do pai
*   - Caso haja cruzamento, cada parametro é cruzado separadamente
*   - O quanto o filho vai herdar de cada pai é escolhido aleatoriamente
*/
function son = crossover(parents)
    probability = grand('def');
    parentA = get_param(parents, 'A');
    parentB = get_param(parents, 'B');
    son = parentA;
    if probability <= crossoverRate then
        kp = getCross(get_param(parentA, 'kp'), get_param(parentB, 'kp'));
        ki = getCross(get_param(parentA, 'ki'), get_param(parentB, 'ki'));
        kd = getCross(get_param(parentA, 'kd'), get_param(parentB, 'kd'));
        son = generateIndividual(kp, ki, kd);
    end
    
endfunction

/*
*   Executa a mutação de um individuo
*   - Determina se é necessário que ocorra a mutação
*   - Verifica em qual parametro deve ocorrer a mesma
*/
function mutated = mutation(individual)
    kp = get_param(individual, 'kp');
    ki = get_param(individual, 'ki');
    kd = get_param(individual, 'kd');
    mutated = individual;
    probability = grand('def');
    if probability <= mutateRate * 1/3 then
        mutated = generateIndividual(-1, ki, kd);
        else if probability <= mutateRate * 2/3 then
            mutated = generateIndividual(kp, -1, kd);
            else if probability <= mutateRate then
                mutated = generateIndividual(kp, ki, -1);
            end
        end
    end

endfunction

/*
*   Retorna uma nova geração de individuos
*   - A nova geração é gerada por meio de:
*        1 - Cruzamento: dois pais são selecionados para o cruzamento
*           onde o filho tem seus parametros compostos por meio de
*           herança de ambos os pais
*        2 - Mutação: O individuo sofre mutação e tem um de seus 
*           parametros alterados de forma aleatória
*        3 - Elitismo: O melhor individuo de cada geração - a elite -
*           permanece inalterado para a próxima geração
*/
function newPopulation = getNewGen(population)
    probability = normalize(getAllFitness(population));
    newPopulation = list();
    newPopulation(1) = getBest(population);
    for i = 2:popSize
        parents = selection(population, probability);
        newPopulation(i) = crossover(parents);
        newPopulation(i) = mutation(newPopulation(i));
    end
endfunction

/*
*    Função que executa todo o processo do algoritmo genetico
*    - Deve ser chamada tendo como parametro a
*       função de transferencia a ser controlada
*/
function PID = geneticAlgorithm(HS)
    population = generatePopInitial(popSize);
    for i = 1:nGenerations
        population = getNewGen(population);
        if modulo(i, 10) == 0 then
            disp(i);
        end
        
    end
    parameters = getBest(population);
    disp(parameters)
    PID = transformToPID(parameters);
endfunction

