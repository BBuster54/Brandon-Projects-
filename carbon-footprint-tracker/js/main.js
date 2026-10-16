import { defaults, validate, calculate, biggestLever } from "./model.js";
import { renderResults, renderMethodology } from "./render.js";
import { readStorage, writeStorage } from "./state.js";
const form=document.querySelector("#footprint-form");
const status=document.querySelector("#load-status");
let data, currentInput, currentResult, timer;
const themeButton=document.querySelector("#theme");
let theme=readStorage("carbon-theme","system");
function applyTheme(){ if(!["light","dark"].includes(theme)) {theme="system";delete document.documentElement.dataset.theme;} else document.documentElement.dataset.theme=theme; themeButton.textContent="Theme: "+theme;}
applyTheme(); themeButton.addEventListener("click",()=>{theme=({system:"light",light:"dark",dark:"system"})[theme];applyTheme();writeStorage("carbon-theme",theme);});
function getInput(){return Object.fromEntries(new FormData(form));}
function setInput(input){for(const key of Object.keys(defaults)) if(form.elements[key]) form.elements[key].value=input[key];}
function update(){
  const input=getInput(), check=validate(input);
  for(const key of Object.keys(defaults)) {
    const control=form.elements[key]; control.setAttribute("aria-invalid",String(Boolean(check.errors[key])));
    const message=document.querySelector("#"+key+"-error");if(message)message.textContent=check.errors[key]||check.warnings[key]||"";
  }
  document.querySelector("#invalid-status").textContent=check.valid?"":"Correct the highlighted inputs. Results show the last valid estimate.";
  if(!check.valid){document.dispatchEvent(new CustomEvent("footprint-invalid"));return;}
  currentInput=input;currentResult=calculate(input,data);
  renderResults(currentResult,input,biggestLever(input,data));
  document.dispatchEvent(new CustomEvent("footprint-update",{detail:{input:currentInput,result:currentResult,data}}));
}
form.addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(update,100);});
form.addEventListener("change",()=>{clearTimeout(timer);update();});
form.addEventListener("submit",event=>{event.preventDefault();clearTimeout(timer);update();});
try {
  const response=await fetch("data/factors.json"); if(!response.ok)throw new Error("Model unavailable");
  data=await response.json();setInput(defaults);renderMethodology(data);status.hidden=true;
  // phase-two-init
  // phase-three-init
  update();
}catch{status.textContent="The emissions model could not load. Reload this page to try again.";form.querySelectorAll("input,select").forEach(c=>c.disabled=true);}

