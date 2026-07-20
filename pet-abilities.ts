import { readFileSync, stat } from "fs";
import { Food, Pack, Pet, State } from "./types/state";
import { chooseRandom, shuffle } from "./utils";
import { createTierUps } from "./environment";

// Abilities kept as a map.
// An ability has a trigger and an effect.

// Actions execute code that loops through state's team's pets
// and applies the effect to the relevant pets.
export enum TRIGGER {
  AFTER_ATTACK = 'After attack',
  BUY = 'Buy',
  BEFORE_ATTACK = 'Before attack',
  EATS_FOOD = 'Eats food',
  END_TURN = 'End turn',
  FAINT = 'Faint',
  FOUR_FRIENDS_HURT = 'Four friends hurt',
  FRIEND_AHEAD_ATTACKS = 'Friend ahead attacks',
  FRIEND_AHEAD_FAINTS = 'Friend ahead faints',
  FRIEND_FAINTS = 'Friend faints',
  FRIEND_SUMMONED = 'Friend summoned',
  FRIENDLY_ATE_FOOD = 'Friendly ate food',
  HURT = 'Hurt',
  KNOCK_OUT = 'Knock out',
  LEVEL_UP = 'Level-up',
  SELL = 'Sell',
  START_OF_BATTLE = 'Start of battle',
  START_OF_TURN = 'Start of turn',
  SUMMONED = 'Summoned',
  TIER_1_FRIEND_BOUGHT = 'Tier 1 friend bought',
  BUY_FOOD = 'buy_food', // Reserved for Cat
}

const rawPets = JSON.parse(readFileSync('pets.json', 'utf-8'));
const rawFood = JSON.parse(readFileSync('food.json', 'utf-8'));
const rawPerks = JSON.parse(readFileSync('perks.json', 'utf-8'));
export let ALL_PETS: {[key: string]: any} = {};
export let ALL_FOOD: any = {};
export let ALL_PERKS: {[key: string]: any} = {};
export let TURTLE_PACK_ABILITY_MAP: any = {};
export let TURTLE_PACK: Pack = {
  pets: [[], [], [], [], [], []],
  food: [[], [], [], [], [], []]
}

let MAX_TRIGGERS: {[key: string]: number} = {
  "60": 3, // Rabbit
  "25": 4, // Dragon
  "11": 2, // Cat
  "38": 3, // Hippo
  "73": 5,  // Snake
  "52": 1,  // Ox
}

for (const pet of rawPets) {
  ALL_PETS[pet.Id] = pet;
  if(pet.Packs.includes("Pack1")){
    let maxTriggers = -1;
    if(pet.Id in MAX_TRIGGERS){
      maxTriggers = MAX_TRIGGERS[(pet.Id as string)];
    }
    TURTLE_PACK.pets[pet.Tier - 1].push({
      id: Number(pet.Id),
      tier: pet.Tier,
      name: pet.Name,
      attack: pet.Attack,
      health: pet.Health,
      tempAttack: 0,
      tempHealth: 0,
      exp: 0,
      perk: "",
      mana: 0,
      maxTriggers: maxTriggers,
      triggersLeft: maxTriggers
    });

    TURTLE_PACK_ABILITY_MAP[pet.Id] = [
      pet.Abilities[0].About.split(":")[0].trim(),
      pet.Abilities[1].About.split(":")[1].trim(),
      pet.Abilities[2].About.split(":")[2].trim(),
    ];
  }
}

for(const perk of rawPerks){
  ALL_PERKS[perk.Id] = perk;
}

TURTLE_PACK_ABILITY_MAP["11"] = TRIGGER.BUY_FOOD; // Cat

export const FOOD_COSTS: {[key: number]: number} = {
  92: 1, // Pill
};

export const FOOD_WITH_NO_TARGET = [
  63, // Pizza
  73, // Salad Bowl
  16, // Canned Food
  82, // Sushi
];


for (const food of rawFood) {
  ALL_FOOD[Number(food.Id)] = food;
  if(food.Packs.includes("Pack1")){
    TURTLE_PACK.food[food.Tier - 1].push({
      id: Number(food.Id),
      name: food.Name,
      frozen: false,
      cost: Number(food.Id) in FOOD_COSTS ? FOOD_COSTS[Number(food.Id)] : 3,
      targetsPet: Number(food.Id) !in FOOD_WITH_NO_TARGET
    });
  }
}

export function activateEatsFood(position: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  newState = activateAbility(TRIGGER.EATS_FOOD, position, newState);
  for(let i = 0; i < newState.team.length; i++){
    if(newState.team[i] !== null){
      newState = activateAbility(TRIGGER.FRIENDLY_ATE_FOOD, i, newState);
    }
  }
  return newState;
};

export function applyPerk(perk: string, position: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  newState.team[position]!.perk = perk;
  activateEatsFood(position, newState);
  return newState;
}

export function removePerk(position: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  newState.team[position]!.perk = "";
  return newState;
}

export function randomTargets(numberOfTargets: number, team: (Pet | null)[], excludePosition: number, filterFunc?: (pet: Pet) => boolean): number[] {
  let targets: number[] = [];
  let availablePositions: number[] = [];
  for(let i = 0; i < team.length; i++){
    if(team[i] !== null && i !== excludePosition){
      if(filterFunc === undefined || filterFunc(team[i]!)){
        availablePositions.push(i);
      }
    }
  }
  for(let i = 0; i < numberOfTargets && availablePositions.length > 0; i++){
    let randomIndex = Math.floor(Math.random() * availablePositions.length);
    targets.push(availablePositions[randomIndex]);
    availablePositions.splice(randomIndex, 1);
  }
  return targets;
}

export function nearestBehind(position: number, team: (Pet | null)[]): number | null {
  for(let i = position - 1; i >= 0; i--){
    if(team[i] !== null){
      return i;
    }
  }
  return null;
}

export function nearestAhead(position: number, team: (Pet | null)[]): number | null {
  for(let i = position + 1; i < team.length; i++){
    if(team[i] !== null){
      return i;
    }
  }
  return null;
}

export function frontmost(team: (Pet | null)[]): number | null {
  for(let i = 0; i < team.length; i++){
    if(team[i] !== null){
      return i;
    }
  }
  return null;
}

export function giveXP(position: number, xp: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  if(newState.team[position] !== null){
    let oldXP = newState.team[position].exp;
    newState.team[position].exp += xp;
    newState.team[position].attack += xp;
    newState.team[position].health += xp;
    if(oldXP < 2 && newState.team[position].exp >= 2){
      newState = activateAbility(TRIGGER.LEVEL_UP, position, newState);
      newState = createTierUps(newState, TURTLE_PACK);
    }
    // This xp < 3 might come back to bite me if I ever have to make Quetzalcoatl
    if(oldXP < 5 && newState.team[position]!.exp >= 5 && xp < 3){
      newState = activateAbility(TRIGGER.LEVEL_UP, position, newState);
      newState = createTierUps(newState, TURTLE_PACK);
    }
  }
  return newState;
}

export function buff(position: number, attackBuff: number, healthBuff: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  if(newState.team[position] !== null){
    newState.team[position].attack += attackBuff;
    newState.team[position].health += healthBuff;
  }
  return newState;
}

export function foodBuff(position: number, attackBuff: number, healthBuff: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  newState = buff(position, attackBuff, healthBuff, newState);
  newState = activateEatsFood(position, newState);
  return newState;
}

export function getLevelFromXP(xp: number): number {
  if(xp >= 5){
    return 3;
  }else if(xp >= 2){
    return 2;
  }else{
    return 1;
  }
}

export function levelToXP(level: number): number {
  if(level >= 3){
    return 5;
  }else if(level === 2){
    return 2;
  }
  return 0;
}

export function activateAbility(trigger: TRIGGER, position: number, state: State, target?: number): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  let pet = state.team[position];
  if(pet !== null){
    let petTrigger = TURTLE_PACK_ABILITY_MAP[pet.id];
    if(petTrigger === trigger && pet.triggersLeft !== 0){
      // Activate ability
      newState = PET_ABILITIES[pet.id](pet, position, getLevelFromXP(pet.exp), newState, target);
      if(pet.maxTriggers > 0){
        pet.triggersLeft--;
      }
    }
  }
  return newState;
}

export function activateAbilitiesInAttackOrder(trigger: TRIGGER, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  let indicesOfPets: number[] = newState.team.filter(pet => pet !== null).map((pet, index) => index);
  shuffle(indicesOfPets);
  indicesOfPets.sort((a, b) => {
    let petA = newState.team[a]!;
    let petB = newState.team[b]!;
    return (petB.attack + petB.tempAttack) - (petA.attack + petA.tempAttack);
  });
  for(let i of indicesOfPets){
    newState = activateAbility(trigger, i, newState);
  }
  return newState;
}

export function summon(position: number, pet: Pet, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  if(newState.team[position] === null){
    // Activate summon abilities
    newState.team[position] = pet;
    newState = activateAbility(TRIGGER.SUMMONED, position, newState);
    for(let i = 0; i < newState.team.length; i++){
      if(newState.team[i] !== null && i !== position){
        newState = activateAbility(TRIGGER.FRIEND_SUMMONED, i, newState, position);
      }
    }
  }else{
    // Find next available position
    for(let i = 0; i < newState.team.length; i++){
      if(newState.team[i] === null){
        newState.team[i] = pet;
        newState = activateAbility(TRIGGER.SUMMONED, i, newState);
        for(let j = 0; j < newState.team.length; j++){
          if(newState.team[j] !== null && j !== i){
            newState = activateAbility(TRIGGER.FRIEND_SUMMONED, j, newState, position);
          }
        }
        break;
      }
    }
  }
  return newState;
}

export function knockOut(position: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  // Activate faint abilities
  let petAtPosition = JSON.parse(JSON.stringify(newState.team[position]));
  newState = activateAbility(TRIGGER.FAINT, position, newState);
  let friendBehind = nearestBehind(position, newState.team);
  if(friendBehind !== null){
    newState = activateAbility(TRIGGER.FRIEND_AHEAD_FAINTS, friendBehind, newState, position);
  }
  for(let i = 0; i < newState.team.length; i++){
    if(newState.team[i] !== null && i !== position){
      newState = activateAbility(TRIGGER.FRIEND_FAINTS, i, newState, position);
    }
  }
  // Remove pet from team
  newState.team[position] = null;
  if(petAtPosition && petAtPosition.perk === "Mushroom"){
    let newPet = JSON.parse(JSON.stringify(petAtPosition));
    newPet.attack = 1;
    newPet.health = 1;
    newPet.tempAttack = 0;
    newPet.tempHealth = 0;
    newState = summon(position, newPet, newState);
  }else if(petAtPosition && petAtPosition.perk === "Honey"){
    let newPet: Pet = {
      id: 4, // Bee
      name: "Bee",
      tier: 1,
      attack: 1,
      health: 1,
      tempAttack: 0,
      tempHealth: 0,
      exp: 0,
      perk: "",
      mana: 0,
      maxTriggers: -1,
      triggersLeft: -1
    };
    newState = summon(position, newPet, newState);
  }
  return newState;
}

export function dealDamage(position: number, damage: number, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  if(newState.team[position] !== null){
    if(newState.team[position].perk === "Garlic"){
      damage = Math.max(2, damage - 2);
    }
    if(newState.team[position].perk === "Melon"){
      damage = Math.max(0, damage - 20);
    }
    if(newState.team[position].perk === "Coconut"){
      damage = 0;
    }
    if(newState.team[position].tempHealth > 0){
      if(damage > newState.team[position].tempHealth){
        damage -= newState.team[position].tempHealth;
        newState.team[position].tempHealth = 0;
      }else{
        newState.team[position].tempHealth -= damage;
        damage = 0;
      }
    }
    newState.team[position].health -= damage;
    if(newState.team[position].perk === "Melon" ||
      newState.team[position].perk === "Coconut"
    ){
      newState = removePerk(position, newState);
    }
    // Activate hurt abilities
    if(damage > 0){
      newState = activateAbility(TRIGGER.HURT, position, newState);
    }
    // Check for knock out
    for(let i = 0; i < newState.team.length; i++){
      if(newState.team[i] !== null && newState.team[i]!.health <= 0){
        // Activate knock out abilities
        newState = knockOut(i, newState);
      }
    }
  }
  return newState;
}

function stockFood(food: Food, state: State): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  newState.shop.food.push(food);
  // TODO: Handle deletion logic
  return newState;
}

export const PET_ABILITIES: {
  [key: number]: (pet: Pet, position: number, lvl: number, state: State, target?: number) => State
} = {
  0: (pet: Pet, position: number, lvl: number, state: State) => { // Ant
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(1, newState.team, position);
    for(let target of targets){
      newState = buff(target, lvl, lvl, newState);
    }
    return newState;
  },
  3: (pet: Pet, position: number, lvl: number, state: State) => { // Beaver
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(2, newState.team, position);
    for(let target of targets){
      newState = buff(target, lvl, 0, newState);
    }
    return newState;
  },
  26: (pet: Pet, position: number, lvl: number, state: State) => { // Duck
    let newState: State = JSON.parse(JSON.stringify(state));
    newState.shop.pets.forEach((shopPet, index) => {
      shopPet.health += lvl;
    });
    return newState;
  },
  559: (pet: Pet, position: number, lvl: number, state: State) => { // Pigeon
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = stockFood({
      id: 139,
      name: "Bread Crumbs",
      frozen: false,
      cost: 0,
      targetsPet: true
    }, newState);
    return newState;
  },
  51: (pet: Pet, position: number, lvl: number, state: State) => { // Otter
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(lvl, newState.team, position);
    for(let target of targets){
      newState = buff(target, 0, 1, newState);
    }
    return newState;
  },
  59: (pet: Pet, position: number, lvl: number, state: State) => { // Pig
    let newState: State = JSON.parse(JSON.stringify(state));
    newState.gold += lvl;
    return newState;
  },
  47: (pet: Pet, position: number, lvl: number, state: State) => { // Mosquito
    // In-battle only
    return state; // TODO: Implement
  },
  32: (pet: Pet, position: number, lvl: number, state: State) => { // Fish
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(2, newState.team, position);
    for(let target of targets){
      newState = buff(target, lvl - 1, lvl - 1, newState);
    }
    return newState;
  },
  17: (pet: Pet, position: number, lvl: number, state: State) => { // Cricket
    let newState: State = JSON.parse(JSON.stringify(state));
    // Summon Zombie Cricket
    newState = summon(position, {
      id: 86,
      name: "Zombie Cricket",
      tier: 1,
      attack: lvl,
      health: lvl,
      tempAttack: 0,
      tempHealth: 0,
      exp: levelToXP(lvl),
      perk: "",
      mana: 0,
      maxTriggers: -1,
      triggersLeft: -1
    }, newState);
    return newState;
  },
  39: (pet: Pet, position: number, lvl: number, state: State, target?: number) => { // Horse
    let newState: State = JSON.parse(JSON.stringify(state));
    if(target){
      newState.team[target]!.tempAttack += lvl;
    }
    return newState;
  },
  72: (pet: Pet, position: number, lvl: number, state: State) => { // Snail
    let newState: State = JSON.parse(JSON.stringify(state));
    for(let i = position; i < Math.min(newState.team.length, position + 3); i++){
      newState = buff(i, lvl, 0, newState);
    }
    return newState;
  },
  16: (pet: Pet, position: number, lvl: number, state: State) => { // Crab
    // In-battle only
    return state; // TODO: Implement
  },
  76: (pet: Pet, position: number, lvl: number, state: State) => { // Swan
    let newState: State = JSON.parse(JSON.stringify(state));
    newState.gold += lvl;
    return newState;
  },
  57: (pet: Pet, position: number, lvl: number, state: State) => { // Rat
    // In-battle only
    return state; // TODO: Implement
  },
  37: (pet: Pet, position: number, lvl: number, state: State) => { // Hedgehog
    let newState: State = JSON.parse(JSON.stringify(state));
    for(let i = 0; i < newState.team.length; i++){
      if(newState.team[i] !== null && i !== position){
        newState = dealDamage(i, lvl * 2, newState);
      }
    }
    return newState;
  },
  54: (pet: Pet, position: number, lvl: number, state: State) => { // Peacock
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = buff(position, lvl * 3, 0, newState);
    return newState;
  },
  29: (pet: Pet, position: number, lvl: number, state: State) => { // Flamingo
    let newState: State = JSON.parse(JSON.stringify(state));
    let nextBehind = nearestBehind(position, newState.team);
    if(nextBehind !== null){
      newState = buff(nextBehind, lvl, lvl, newState);
      let nextNextBehind = nearestBehind(nextBehind, newState.team);
      if(nextNextBehind !== null){
        newState = buff(nextNextBehind, lvl, lvl, newState);
      }
    }
    return newState;
  },
  40: (pet: Pet, position: number, lvl: number, state: State) => { // Kangaroo
    // In-battle only
    return state; // TODO: Implement
  },
  74: (pet: Pet, position: number, lvl: number, state: State) => { // Spider
    let newState: State = JSON.parse(JSON.stringify(state));
    let newPet = chooseRandom(TURTLE_PACK.pets[2]);
    newPet.attack = lvl * 2;
    newPet.health = lvl * 2;
    newPet.exp = levelToXP(lvl);
    newState = summon(position, newPet, newState);
    return newState;
  },
  21: (pet: Pet, position: number, lvl: number, state: State) => { // Dodo
    // In-battle only
    return state; // TODO: Implement
  },
  2: (pet: Pet, position: number, lvl: number, state: State) => { // Badger
    let newState: State = JSON.parse(JSON.stringify(state));
    if(position > 0){
      newState = dealDamage(position - 1, Math.ceil(0.5 * lvl * (pet.attack + pet.tempAttack)), newState);
    }
    if(position < newState.team.length - 1){
      newState = dealDamage(position + 1, Math.ceil(0.5 * lvl * (pet.attack + pet.tempAttack)), newState);
    }
    return newState;
  },
  23: (pet: Pet, position: number, lvl: number, state: State) => { // Dolphin
    // In-battle only
    return state; // TODO: Implement
  },
  33: (pet: Pet, position: number, lvl: number, state: State) => { // Giraffe
    let newState: State = JSON.parse(JSON.stringify(state));
    let currentPosition = nearestAhead(position, newState.team);
    for(let i = 0; i < lvl; i++){
      if(currentPosition !== null){
        newState = buff(currentPosition, 1, 1, newState);
        currentPosition = nearestAhead(currentPosition, newState.team);
      }
    }
    return newState;
  },
  28: (pet: Pet, position: number, lvl: number, state: State) => { // Elephant
    // In-battle only
    return state; // TODO: Implement
  },
  10: (pet: Pet, position: number, lvl: number, state: State) => { // Camel
    let newState: State = JSON.parse(JSON.stringify(state));
    let nextBehind = nearestBehind(position, newState.team);
    if(nextBehind !== null){
      newState = buff(nextBehind, lvl, lvl * 2, newState);
    }
    return newState;
  },
  60: (pet: Pet, position: number, lvl: number, state: State, target?: number) => { // Rabbit
    let newState: State = JSON.parse(JSON.stringify(state));
    if(target){
      newState = buff(target, 0, lvl, newState);
    }
    return newState;
  },
  52: (pet: Pet, position: number, lvl: number, state: State) => { // Ox
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Melon", position, newState);
    newState = buff(position, 1, 0, newState);
    return newState;
  },
  22: (pet: Pet, position: number, lvl: number, state: State) => { // Dog
    let newState: State = JSON.parse(JSON.stringify(state));
    newState.team[position]!.tempAttack += lvl * 2;
    newState.team[position]!.health += lvl;
    return newState;
  },
  68: (pet: Pet, position: number, lvl: number, state: State) => { // Sheep
    let newState: State = JSON.parse(JSON.stringify(state));
    // Summon Rams
    for(let i = 0; i < 2; i++){
      newState = summon(position, {
        id: 62,
        name: "Ram",
        tier: 1,
        attack: lvl * 2,
        health: lvl * 2,
        tempAttack: 0,
        tempHealth: 0,
        exp: levelToXP(lvl),
        perk: "",
        mana: 0,
        maxTriggers: -1,
        triggersLeft: -1
      }, newState);
    }
    return newState;
  },
  70: (pet: Pet, position: number, lvl: number, state: State) => { // Skunk
    // In-battle only
    return state; // TODO: Implement
  },
  38: (pet: Pet, position: number, lvl: number, state: State) => { // Hippo
    // In-battle only
    return state; // TODO: Implement
  },
  5: (pet: Pet, position: number, lvl: number, state: State) => { // Bison
    let newState: State = JSON.parse(JSON.stringify(state));
    if(state.team.some(pet => pet?.exp === 5)){
      newState = buff(position, lvl, lvl * 2, newState);
    }
    return newState;
  },
  7: (pet: Pet, position: number, lvl: number, state: State) => { // Blowfish
    // In-battle only
    return state; // TODO: Implement
  },
  80: (pet: Pet, position: number, lvl: number, state: State) => { // Turtle
    let newState: State = JSON.parse(JSON.stringify(state));
    let nearestPetBehind = nearestBehind(position, newState.team);
    if(nearestPetBehind !== null){
      newState = applyPerk("Melon", nearestPetBehind, newState);
    }
    return newState;
  },
  75: (pet: Pet, position: number, lvl: number, state: State) => { // Squirrel
    let newState: State = JSON.parse(JSON.stringify(state));
    for(let food of newState.shop.food){
      food.cost = Math.max(0, food.cost - lvl);
    }
    return newState;
  },
  56: (pet: Pet, position: number, lvl: number, state: State) => { // Penguin
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(2, newState.team, position, (pet) => getLevelFromXP(pet.exp) > 1);
    for(let target of targets){
      newState = buff(target, lvl, lvl, newState);
    }
    return newState;
  },
  20: (pet: Pet, position: number, lvl: number, state: State) => { // Deer
    let newState: State = JSON.parse(JSON.stringify(state));
    // Summon bus
    newState = summon(position, {
      id: 85,
      name: "Bus",
      tier: 1,
      attack: 5 * lvl,
      health: 3 * lvl,
      tempAttack: 0,
      tempHealth: 0,
      exp: levelToXP(lvl),
      perk: "Chili",
      mana: 0,
      maxTriggers: -1,
      triggersLeft: -1
    }, newState);
    return newState;
  },
  81: (pet: Pet, position: number, lvl: number, state: State) => { // Whale
    // In-battle only
    return state; // TODO: Implement
  },
  53: (pet: Pet, position: number, lvl: number, state: State) => { // Parrot
    // In-battle only
    // This fucker is going to require so much damn work
    // I need to be able to track abilities on pets!
    // Fortunately none of that tech matters for turtle
    return state; // TODO: Implement
  },
  65: (pet: Pet, position: number, lvl: number, state: State) => { // Scorpion
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Peanut", position, newState);
    return newState;
  },
  19: (pet: Pet, position: number, lvl: number, state: State) => { // Crocodile
    // In-battle only
    return state; // TODO: Implement
  },
  55: (pet: Pet, position: number, lvl: number, state: State) => { // Rhino
    // In-battle only
    return state; // TODO: Implement
  },
  46: (pet: Pet, position: number, lvl: number, state: State) => { // Monkey
    let newState: State = JSON.parse(JSON.stringify(state));
    let frontPet = frontmost(newState.team);
    if(frontPet !== null){
      newState = buff(frontPet, 2 * lvl, 2 * lvl, newState);
    }
    return newState;
  },
  166: (pet: Pet, position: number, lvl: number, state: State) => { // Armadillo
    // In-battle only
    return state; // TODO: Implement
  },
  15: (pet: Pet, position: number, lvl: number, state: State) => { // Cow
    let newState: State = JSON.parse(JSON.stringify(state));
    let milk: Food;
    switch(lvl){
      case 3:
        milk = {
          id: 103,
          name: "Best Milk",
          cost: 0,
          frozen: false,
          targetsPet: true
        };
        break;
      case 2:
        milk = {
          id: 102,
          name: "Better Milk",
          cost: 0,
          frozen: false,
          targetsPet: true
        };
        break;
      default:
        milk = {
          id: 49,
          name: "Milk",
          cost: 0,
          frozen: false,
          targetsPet: true
        };
    };
    newState = stockFood(milk, newState);
    return newState;
  },
  66: (pet: Pet, position: number, lvl: number, state: State) => { // Seal
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(3, newState.team, position);
    for(let target of targets){
      newState = buff(target, lvl, 0, newState);
    }
    return newState;
  },
  63: (pet: Pet, position: number, lvl: number, state: State) => { // Rooster
    let newState: State = JSON.parse(JSON.stringify(state));
    // Summon chick(s)
    for(let i = 0; i < lvl; i++){
      newState = summon(position, {
        id: 13,
        name: "Chick",
        tier: 1,
        attack: Math.ceil(pet.attack/2),
        health: 1,
        tempAttack: 0,
        tempHealth: 0,
        exp: levelToXP(lvl),
        perk: "",
        mana: 0,
        maxTriggers: -1,
        triggersLeft: -1
      }, newState);
    }
    return newState;
  },
  69: (pet: Pet, position: number, lvl: number, state: State) => { // Shark
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = buff(position, lvl * 2, lvl * 2, newState);
    return newState;
  },
  79: (pet: Pet, position: number, lvl: number, state: State, target?: number) => { // Turkey
    let newState: State = JSON.parse(JSON.stringify(state));
    if(target){
      newState = buff(target, lvl * 3, lvl, newState);
    }
    return newState;
  },
  41: (pet: Pet, position: number, lvl: number, state: State) => { // Leopard
    // In-battle only
    return state; // TODO: Implement
  },
  103: (pet: Pet, position: number, lvl: number, state: State) => { // Boar
    // In-battle only
    return state; // TODO: Implement
  },
  77: (pet: Pet, position: number, lvl: number, state: State) => { // Tiger
    // In-battle only
    return state; // TODO: Implement
  },
  269: (pet: Pet, position: number, lvl: number, state: State) => { // Wolverine
    // In-battle only
    return state; // TODO: Implement
  },
  36: (pet: Pet, position: number, lvl: number, state: State) => { // Gorilla
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Coconut", position, newState);
    return newState;
  },
  25: (pet: Pet, position: number, lvl: number, state: State) => { // Dragon
    let newState: State = JSON.parse(JSON.stringify(state));
    for(let i = 0; i < newState.team.length; i++){
      if(newState.team[i] !== null && i !== position){
        newState = buff(i, lvl, lvl, newState);
      }
    }
    return newState;
  },
  45: (pet: Pet, position: number, lvl: number, state: State) => { // Mammoth
    let newState: State = JSON.parse(JSON.stringify(state));
    for(let i = 0; i < newState.team.length; i++){
      if(newState.team[i] !== null && i !== position){
        newState = buff(i, lvl * 2, lvl * 2, newState);
      }
    }
    return newState;
  },
  // Cat will be implemented in the Buy Food action :(
  73: (pet: Pet, position: number, lvl: number, state: State) => { // Snake
    // In-battle only
    return state; // TODO: Implement
  },
  30: (pet: Pet, position: number, lvl: number, state: State) => { // Fly
    let newState: State = JSON.parse(JSON.stringify(state));
    // Summon zombie fly
    for(let i = 0; i < lvl; i++){
      newState = summon(position, {
        id: 31,
        name: "Zombie Fly",
        tier: 1,
        attack: 4 * lvl,
        health: 4 * lvl,
        tempAttack: 0,
        tempHealth: 0,
        exp: levelToXP(lvl),
        perk: "",
        mana: 0,
        maxTriggers: -1,
        triggersLeft: -1
      }, newState);
    }
    return newState;
  }
};

export const FOOD_ABILITIES: {
  [key: number]: (targetPosition: number, state: State) => State
} = {
  0: (targetPosition: number, state: State) => { // Apple
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 1, 1, newState);
    return newState;
  },
  40: (targetPosition: number, state: State) => { // Honey
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Honey", targetPosition, newState);
    return newState;
  },
  92: (targetPosition: number, state: State) => { // Pill
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = knockOut(targetPosition, newState);
    return newState;
  },
  9: (targetPosition: number, state: State) => { // Meat Bone
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Meat Bone", targetPosition, newState);
    return newState;
  },
  50: (targetPosition: number, state: State) => { // Cupcake
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 0, 0, newState);
    newState.team[targetPosition]!.tempAttack += 3;
    newState.team[targetPosition]!.tempHealth += 3;
    return newState;
  },
  73: (targetPosition: number, state: State) => { // Salad Bowl
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(2, newState.team, -1);
    for(let target of targets){
      newState = foodBuff(target, 1, 1, newState);
    }
    return newState;
  },
  38: (targetPosition: number, state: State) => { // Garlic
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Garlic", targetPosition, newState);
    return newState;
  },
  16: (targetPosition: number, state: State) => { // Canned Food
    let newState: State = JSON.parse(JSON.stringify(state));
    newState.shopAttack += 1;
    newState.shopHealth += 1;
    return newState;
  },
  58: (targetPosition: number, state: State) => { // Pear
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 2, 2, newState);
    return newState;
  },
  22: (targetPosition: number, state: State) => { // Chili
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Chili", targetPosition, newState);
    return newState;
  },
  82: (targetPosition: number, state: State) => { // Sushi
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(3, newState.team, -1);
    for(let target of targets){
      newState = foodBuff(target, 1, 1, newState);
    }
    return newState;
  },
  23: (targetPosition: number, state: State) => { // Chocolate
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = giveXP(targetPosition, 1, newState);
    newState = activateEatsFood(targetPosition, newState);
    return newState;
  },
  79: (targetPosition: number, state: State) => { // Steak
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Steak", targetPosition, newState);
    return newState;
  },
  96: (targetPosition: number, state: State) => { // Melon
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Melon", targetPosition, newState);
    return newState;
  },
  51: (targetPosition: number, state: State) => { // Mushroom
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = applyPerk("Mushroom", targetPosition, newState);
    return newState;
  },
  63: (targetPosition: number, state: State) => { // Pizza
    let newState: State = JSON.parse(JSON.stringify(state));
    let targets = randomTargets(2, newState.team, -1);
    for(let target of targets){
      newState = foodBuff(target, 2, 2, newState);
    }
    return newState;
  },
  49: (targetPosition: number, state: State) => { // Milk
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 1, 2, newState);
    return newState;
  },
  102: (targetPosition: number, state: State) => { // Better Milk
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 2, 4, newState);
    return newState;
  },
  103: (targetPosition: number, state: State) => { // Best Milk
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 3, 6, newState);
    return newState;
  },
  139: (targetPosition: number, state: State) => { // Bread Crumbs
    let newState: State = JSON.parse(JSON.stringify(state));
    newState = foodBuff(targetPosition, 1, 0, newState);
    return newState;
  }
}