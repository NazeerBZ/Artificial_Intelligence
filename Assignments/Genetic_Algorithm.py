from random import randint, random
from operator import add

fitness_history = []

def generate_individual(individualSize, minVal, maxVal):
    randVal = randint(minVal, maxVal)
    randVal_bin = format(randVal,'b')
    binaryForm = []
    for i in range(len(randVal_bin)):
        binaryForm.append(int(randVal_bin[i]))

    zeros_len = individualSize - len(randVal_bin)
    zeros = []
    for x in range(zeros_len):
        zeros.append(0)
    if len(binaryForm) != individualSize:
        return zeros + binaryForm
    else:
        return binaryForm

def generate_population(populationSize, individualSize, minVal, maxVal):    
    if populationSize >= 10:
        return [generate_individual(individualSize, minVal, maxVal) for x in range(populationSize)]
    else:
        return 'population size must be equal or greater then 10'
    
def fitness(individual):
    value = ''
    for x in range(len(individual)):
        value = value + str(individual[x])
        
    value = int(value,2)
    return value**2

def selection(popul, retain=0.2, random_select=0.8):
    grading = [(fitness(x), x) for x in popul]
    grading = [x[1] for x in sorted(grading)]
    retain_length = int(len(popul) * retain)
    parents = grading[:retain_length] 
    
    for individual in grading[retain_length:]:
        if random_select > random():
            parents.append(individual)
    return parents         
        
def crossover(popul, parents):
    child_population = parents
    parents_length = len(parents)
    desired_length = len(popul) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:int(half)] + female[int(half):]
            children.append(child)        
    child_population.extend(children)
    return child_population
    
def mutation(child, mutate=0.8):
    counter = 0
    for individual in child:
        counter +=1
        if mutate > random():
            index = randint(0, len(individual)-1)
            individual[index] = 1
            child[counter-1] = individual
    return child

def average_fitness(popul):
    sumValues = 0
    for individual in popul:
        binary = ''
        for val in individual:
            binary = binary + str(val)
        sumValues = sumValues + int(binary,2)
    
    fitness_history.append(sumValues / len(popul))
    
p = generate_population(10,5,0,32)
average_fitness(p)

for x in range(5):
    parent_population = selection(p)
    child_population = crossover(p,parent_population)
    child_population = mutation(child_population)
    average_fitness(child_population)
    p = child_population    

print(fitness_history)