import { chooseRandom } from "../utils"
import { Action } from "./actions"

export interface CalculatorPet {
  name: string,
  attack: number,
  health: number,
  exp: number,
  perk: string,
  mana: number,
  belugaSwallowedPet?: string,
  abominationSwallowedPet1?: string,
  abominationSwallowedPet2?: string,
  abominationSwallowedPet3?: string
}

export interface Pet extends CalculatorPet {
  id: number,
  tier: number,
  tempAttack: number,
  tempHealth: number,
  maxTriggers: number,
  triggersLeft: number
}

export interface ShopPet extends Pet {
  frozen: boolean,
  linkedWith?: number
}

export interface Food {
  id: number,
  name: string,
  frozen: boolean,
  cost: number,
  targetsPet: boolean
}

export interface Shop {
  pets: ShopPet[],
  food: Food[]
}

export interface Pack {
  pets: Pet[][],
  food: Food[][]
}

export interface State {
  team: (Pet | null)[],
  shop: Shop,
  lives: number,
  turnNumber: number,
  gold: number,
  trophies: number,
  shopAttack: number,
  shopHealth: number
}

export interface CalculatorInput {
  playerPack: string,
  opponentPack: string,
  playerToy: string,
  playerToyLevel: number,
  opponentToy: string,
  opponentToyLevel: number,
  turn: number,
  playerGoldSpent: number,
  opponentGoldSpent: number,
  playerRollAmount: number,
  opponentRollAmount: number,
  playerSummonedAmount: number,
  opponentSummonedAmount: number,
  playerLevel3Sold: number,
  opponentLevel3Sold: number,
  playerTransformationAmount: number,
  opponentTransformationAmount: number,
  playerPets: (CalculatorPet | null)[],
  opponentPets: (CalculatorPet | null)[],
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
}