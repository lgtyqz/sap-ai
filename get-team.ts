import { Database } from "sqlite";
import { ALL_PERKS, ALL_PETS } from "./pet-abilities";
import { Pet } from "./types/state";
function getPetInfo(petJSON: any): Pet {
  const petId = Number(petJSON["Enu"] ?? 0);
  const petName = ALL_PETS[petId] ? ALL_PETS[petId].Name : "Token Pet";
  const petLevel = petJSON["Lvl"];
  const petExperience = petJSON["Exp"] ?? 0;
  const petAtk = petJSON["At"]["Perm"];
  const petHp = petJSON["Hp"]["Perm"];
  const petTempAtk = petJSON["At"]["Temp"] ?? 0;
  const petTempHp = petJSON["Hp"]["Temp"] ?? 0;
  const petPerkId = petJSON["Perk"] ?? -1;
  const perkName = petPerkId !== -1 && ALL_PERKS[petPerkId] ? ALL_PERKS[petPerkId].Name : "";
  return {
    id: petId,
    name: petName,
    attack: petAtk,
    health: petHp,
    tempAttack: petTempAtk,
    tempHealth: petTempHp,
    exp: petExperience,
    perk: perkName,
    mana: 0,
    tier: ALL_PETS[petId].Tier,
    maxTriggers: -1,
    triggersLeft: -1
  };
}

function getBoardInfo(boardJSON: any) {
  let board: Pet[] = [];
  for (const petJSON of boardJSON["Mins"]["Items"]) {
    if (petJSON !== null) {
      board.push(getPetInfo(petJSON));
    }
  }

  // Implementing toys later
  // for (const toy of battle["UserBoard"]["Rel"]["Items"]) {
  //   if (toy !== null && toy["Enu"]) {
  //     const toyId = toy["Enu"];
  //     newBattle.playerBoard.toy.imagePath = TOYS[toyId] ? `Sprite/Toys/${TOYS[toyId].NameId}.png` : PLACEHOLDER_SPRITE;
  //     newBattle.playerBoard.toy.level = toy["Lvl"];
  //   }
  // }

  return board;
}

export async function getRandomTeamFromDB(db: Database): Promise<Pet[]> {
  let rawBoard = await db.get("SELECT player_board FROM boards ORDER BY RANDOM() LIMIT 1");
  // Parse SAP API stuff
  return getBoardInfo(JSON.parse(rawBoard.player_board));
}