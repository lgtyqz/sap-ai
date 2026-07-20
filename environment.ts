import { sqlite3 } from "sqlite3";
import { activateAbilitiesInAttackOrder, activateAbility, FOOD_ABILITIES, FOOD_COSTS, FOOD_WITH_NO_TARGET, getLevelFromXP, giveXP, TRIGGER } from "./pet-abilities";
import { Action, ActionType } from "./types/actions";
import { Pack, Shop, Pet, Food, State, ShopPet, CalculatorInput } from "./types/state";
import { chooseRandom, shuffle } from "./utils";
import { Database } from "sqlite";
import { runSimulation, SimulationResult } from "sap-calculator";
import { getRandomTeamFromDB } from "./get-team";

export let ALL_POSSIBLE_REPOSITIONS: number[][] = [];

let originalPositioning = [0, 1, 2, 3, 4];
function generateRepositions(currentPositioning: number[]): number[][] {
  if(currentPositioning.length === 1){
    return [currentPositioning];
  }
  return currentPositioning
    .flatMap((first, index) => {
      let remaining = [
        ...currentPositioning.slice(0, index),
        ...currentPositioning.slice(index + 1, currentPositioning.length)
      ];
      return generateRepositions(remaining)
        .map((positioning) => [first, ...positioning]);
    });
}

ALL_POSSIBLE_REPOSITIONS = generateRepositions(originalPositioning);

export function rollShop(currentShop: Shop, turnNumber: number, state: State, pack: Pack): Shop {
  // Check for linked pets, unlink them if need be
  for(let i = 0; i < currentShop.pets.length; i++){
    if(currentShop.pets[i].linkedWith && currentShop.pets[i].frozen){
      if(!currentShop.pets[currentShop.pets[i].linkedWith!].frozen){
        currentShop.pets[i].linkedWith = undefined;
      }
    }
  }

  let newPets: ShopPet[] = currentShop.pets.filter(pet => pet.frozen);
  let newFood: Food[] = currentShop.food.filter(food => food.frozen);

  let tier = Math.min(Math.floor((turnNumber - 1) / 2) + 1, 6);

  let shopPetSlots = Math.min(3 + Math.floor((tier - 1)/2), 5) - newPets.length;
  let shopFoodSlots = (tier < 3 ? 1 : 2) - newFood.length;
  for(let i = 0; i < shopPetSlots; i++){
    let possiblePets = pack.pets.slice(0, tier).flat();
    let newPet: ShopPet = {
      ...chooseRandom(possiblePets),
      frozen: false
    };
    newPet.attack += state.shopAttack;
    newPet.health += state.shopHealth;
    newPets.push(newPet);
  }
  for(let i = 0; i < shopFoodSlots; i++){
    let possibleFood = pack.food.slice(0, tier).flat();
    newFood.push(
      chooseRandom(possibleFood)
    );
  }

  return {
    pets: newPets,
    food: newFood
  }
}

export function createTierUps(state: State, pack: Pack): State {
  let newState: State = JSON.parse(JSON.stringify(state));
  // Create tierups in shop (deleting slots if necessary)
  // Select two from next tier
  let nextTierIndex = Math.max(Math.floor(newState.turnNumber/2) + 1, 5);
  let nextTierPets = (JSON.parse(JSON.stringify(pack.pets[nextTierIndex])) as Pet[]);
  shuffle(nextTierPets);
  let tierUpPets = nextTierPets.slice(0, 2);
  newState.shop.pets.splice(0, 0, {
    ...tierUpPets[0],
    frozen: false,
    linkedWith: 1
  }, {
    ...tierUpPets[1],
    frozen: false,
    linkedWith: 0
  });

  return newState;
}

export async function newStateFromAction(currentState: State, action: Action, pack: Pack, dbConnection: Database): Promise<State> {
  let newState: State = JSON.parse(JSON.stringify(currentState));
  // Reorder pets
  let newTeam: (Pet | null)[] = [null, null, null, null, null];
  for(let i = 0; i < currentState.team.length; i++){
    newTeam[action.reposition[i]] = currentState.team[i];
  }

  // Perform action
  switch(action.actionType){
    case ActionType.BUY_FOOD:
      // Activate buy food ability
      newState = FOOD_ABILITIES[currentState.shop.food[action.actionTarget].id](action.toPosition!, newState);
      for(let i = 0; i < newState.team.length; i++){
        if(newState.team[i]?.id === 11){ // Cat
          newState = FOOD_ABILITIES[currentState.shop.food[action.actionTarget].id](action.toPosition!, newState);
        }
      }
      newState.gold -= currentState.shop.food[action.actionTarget].cost;
      newState.shop.food.splice(action.actionTarget, 1);
      break;

    case ActionType.BUY_MERGE_PET:
      // Activate buy ability
      newState.team[action.toPosition!]!.attack = Math.max(
        newState.team[action.toPosition!]!.attack,
        newState.shop.pets[action.actionTarget].attack
      );
      newState.team[action.toPosition!]!.health = Math.max(
        newState.team[action.toPosition!]!.health,
        newState.shop.pets[action.actionTarget].health
      );
      newState = giveXP(action.toPosition!, 1, newState);
      newState = activateAbility(TRIGGER.BUY, action.toPosition!, newState);
      if(newState.team[action.toPosition!]!.tier === 1){
        for(let i = 0; i < newState.team.length; i++){
          activateAbility(TRIGGER.TIER_1_FRIEND_BOUGHT, i, newState);
        }
      }
      newState.gold -= 3;
      newState.shop.pets.splice(action.actionTarget, 1);
      break;

    case ActionType.BUY_PET:
      newState.team[action.toPosition!] = newState.shop.pets[action.actionTarget];
      // Activate buy ability
      activateAbility(TRIGGER.BUY, action.toPosition!, newState);
      if(newState.team[action.toPosition!]!.tier === 1){
        for(let i = 0; i < newState.team.length; i++){
          activateAbility(TRIGGER.TIER_1_FRIEND_BOUGHT, i, newState);
        }
      }
      newState.gold -= 3;
      newState.shop.pets.splice(action.actionTarget, 1);
      break;

    case ActionType.FREEZE_FOOD:
      newState.shop.food[action.actionTarget].frozen = true;
      break;

    case ActionType.UNFREEZE_FOOD:
      newState.shop.food[action.actionTarget].frozen = false;
      break;

    case ActionType.UNFREEZE_PET:
      newState.shop.pets[action.actionTarget].frozen = false;
      break;
    
    case ActionType.FREEZE_PET:
      newState.shop.pets[action.actionTarget].frozen = true;
      break;
    
    case ActionType.MERGE_BOARD_PET:
      // Check for level up
      newState.team[action.toPosition!]!.attack = Math.max(
        newState.team[action.toPosition!]!.attack,
        newState.team[action.actionTarget]!.attack
      );
      newState.team[action.toPosition!]!.health = Math.max(
        newState.team[action.toPosition!]!.health,
        newState.team[action.actionTarget]!.health
      );
      newState = giveXP(action.toPosition!, newState.team[action.actionTarget]!.exp + 1, newState);
      newState.team[action.actionTarget!] = null;
      break;

    case ActionType.ROLL_SHOP:
      newState.shop = rollShop(currentState.shop, currentState.turnNumber, newState, pack);
      // Activate roll abilities (none in p1)
      newState.gold -= 1;
      break;

    case ActionType.SELL_PET:
      // Activate sell ability
      newState = activateAbility(TRIGGER.SELL, action.toPosition!, newState);
      newState.gold += getLevelFromXP(newState.team[action.toPosition!]!.exp);
      newState.team[action.toPosition!] = null;
      break;

    case ActionType.END_TURN:
      // Activate end turn abilities
      newState = activateAbilitiesInAttackOrder(TRIGGER.END_TURN, newState);

      // Select random team from turn to face
      let enemyTeam = await getRandomTeamFromDB(dbConnection);
      let simulationConfig: CalculatorInput = {
        playerPack: "Turtle",
        opponentPack: "Turtle",
        // This is turtle pack none of this shit matters lol
        playerToy: "",
        playerToyLevel: 0,
        opponentToy: "",
        opponentToyLevel: 0,
        turn: newState.turnNumber,
        playerGoldSpent: 10,
        opponentGoldSpent: 10,
        playerRollAmount: 0,
        opponentRollAmount: 0,
        playerSummonedAmount: 0,
        opponentSummonedAmount: 0,
        playerLevel3Sold: 0,
        opponentLevel3Sold: 0,
        playerTransformationAmount: 0,
        opponentTransformationAmount: 0,
        // Always relevant gg
        playerPets: newState.team,
        opponentPets: enemyTeam,
        // Default UI settings for a clean calculator state
        angler: false,
        allPets: false,
        logFilter: null,
        fontSize: 13,
        customPacks: [],
        oldStork: false,
        tokenPets: false,
        komodoShuffle: false,
        mana: true,
        showAdvanced: true,
        ailmentEquipment: false
      };
      // Battle!
      let outcome: SimulationResult = runSimulation({
        ...simulationConfig,
        simulationCount: 1
      });
      // Handle lives accordingly
      if(outcome.playerWins === 1){
        newState.trophies += 1;
      }else if(outcome.opponentWins === 1){
        newState.lives -= 1;
      }
      // Reset gold
      newState.turnNumber += 1;
      if(newState.turnNumber === 3 && newState.lives < 5){
        // Bonus life if lost
        newState.lives += 1;
      }
      newState.gold = 10;
      newState.shop = rollShop(currentState.shop, newState.turnNumber, newState, pack);

      // Activate start of turn abilities
      activateAbilitiesInAttackOrder(TRIGGER.START_OF_TURN, newState);
      break;
  }

  return newState;
}

export function newActionsFromState(currentState: State): Action[] {
  let newActions: Action[] = [];
  // For each repositioning, assign an action and a target
  for(let position of ALL_POSSIBLE_REPOSITIONS){
    // Actions available for each shop pet
    for(let shopIndex = 0; shopIndex < currentState.shop.pets.length; shopIndex++){
      // If empty space, buy pet is an option for every pet in shop
      if(currentState.team.includes(null) && currentState.gold >= 3){
        newActions.push({
          actionType: ActionType.BUY_PET,
          reposition: position,
          actionTarget: shopIndex
        });
      }
      // If a pet in shop matches a pet on team, merge pet is an option
      for(let teamIndex = 0; teamIndex < currentState.team.length; teamIndex++){
        if(currentState.team[teamIndex] !== null &&
          currentState.shop.pets[shopIndex].id === currentState.team[teamIndex]!.id &&
          currentState.gold >= 3){
          newActions.push({
            actionType: ActionType.BUY_MERGE_PET,
            reposition: position,
            actionTarget: shopIndex,
            toPosition: position[teamIndex]
          });
        }
      }

      // Might have to disable lol
      if(!currentState.shop.pets[shopIndex].frozen){
        newActions.push({
          actionType: ActionType.FREEZE_PET,
          reposition: position,
          actionTarget: shopIndex
        });
      }else{
        newActions.push({
          actionType: ActionType.UNFREEZE_PET,
          reposition: position,
          actionTarget: shopIndex
        });
      }
    }

    // Actions available for each shop food
    for(let shopIndex = 0; shopIndex < currentState.shop.food.length; shopIndex++){
      let currentFood = currentState.shop.food[shopIndex];
      if(currentState.gold >= currentFood.cost){
        if(FOOD_WITH_NO_TARGET.includes(currentFood.id)){
          newActions.push({
            actionType: ActionType.BUY_FOOD,
            reposition: position,
            actionTarget: -1
          });
        }else{
          for(let teamIndex = 0; teamIndex < currentState.team.length; teamIndex++){
            newActions.push({
              actionType: ActionType.BUY_FOOD,
              reposition: position,
              actionTarget: position[teamIndex]
            });
          }
        }
      }
    }

    // Actions for team
    for(let teamIndex = 0; teamIndex < currentState.team.length; teamIndex++){
      let currentPet = currentState.team[teamIndex];
      if(currentPet !== null){
        newActions.push({
          actionType: ActionType.SELL_PET,
          reposition: position,
          actionTarget: position[teamIndex]
        });

        for(let i = 0; i < currentState.team.length; i++){
          if(currentState.team[i] && currentPet.id === currentState.team[i]!.id){
            newActions.push({
              actionType: ActionType.MERGE_BOARD_PET,
              reposition: position,
              actionTarget: position[teamIndex],
              toPosition: position[i]
            })
          }
        }
      }
    }

    if(currentState.gold >= 1){
      newActions.push({
        actionType: ActionType.ROLL_SHOP,
        reposition: position,
        actionTarget: -1
      });
    }

    // Can always end turn
    newActions.push({
      actionType: ActionType.END_TURN,
      reposition: position,
      actionTarget: -1
    });
  }
  return newActions;
}