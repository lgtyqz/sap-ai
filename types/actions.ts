export enum ActionType {
  BUY_PET = 0,
  BUY_MERGE_PET = 1,
  MERGE_BOARD_PET = 2,
  BUY_FOOD = 3,
  ROLL_SHOP = 4,
  FREEZE_PET = 5,
  FREEZE_FOOD = 6,
  SELL_PET = 7,
  UNFREEZE_PET = 8,
  UNFREEZE_FOOD = 9,
  END_TURN = 10
}

export interface Action {
  actionType: ActionType,
  reposition: number[],
  actionTarget: number,
  toPosition?: number
}