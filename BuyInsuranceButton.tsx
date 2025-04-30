"use client"

import { useState } from "react";
import { ArrowRight } from "lucide-react";

export default function BuyInsuranceButton() {
  const [selectedOption, setSelectedOption] = useState<string>("");

  const handleBuyInsurance = () => {
    // Redirect based on the selected option
    let url = "";
    switch (selectedOption) {
      case "Aetna Medical Insurance":
        url = "https://enrollmedicare.aetna.com/s/shop?tfn=&ZipCode=60607&CountyFIPS=17031&PlanYear=2025&step=PlanList"; // Link for "One"
        break;
      case "Blue Cross and Blue Shield":
        url = "https://www.bcbsil.com/medicare"; // Link for "Two"
        break;
      case "Humana Medical Insurance":
        url = "https://shop.humana-medicareadvantage.com/?pspt=4ce0d070-2601-11f0-b20c-0db53d0967c7&tfn=800-395-8346&app=TZINS10&siteleadid=cb26373f-32e3-4852-83d3-52d246e25179&matchstatus=Unmatched&utm_medium=cpc&utm_source=google&utm_campaign=google&gclid=Cj0KCQjwlMfABhCWARIsADGXdy8kR5rBNJ6jhaoPSGjSrYE_nEOoKyYKhBN3NHsMElgzZS1uQ4PY36kaAuVYEALw_wcB#/plans/60608/17031/MAPD"; // Link for "Three"
        break;
      case "Cigna":
        url = "https://plans.cigna.com/?zip=60608&fip=17031&PlanType=MAPD&customer_id=202-369-9026&utm_campaign=0316145&utm_source=Search&campaign_ID=0316145&utm_medium=Search&sid=0316145&PID=ps_17_25447&customtrack1=0316145&gad_source=1&gbraid=0AAAAADuABamTciPq42pd_5qJiYhFNVD9e&gclid=Cj0KCQjwlMfABhCWARIsADGXdy_shDw3jtGMRIu4QOuMfSZWRGngNhF9KbPLFI-gQHTubywndFL321waAjm3EALw_wcB&gclsrc=aw.ds"; // Link for "Four"
        break;
      case "AARP":
        url = "https://www.aarpmedicareplans.com/health-plans/plan-summary/60608/031/2025#MA"; // Link for "Five"
        break;
      case "Medica":
        url = "https://medica.isf.io/2025/g/7876cf047cb74e33aaecd5e44ff17615/AssistedShopping?step=3";
        break;
      case "Wellcare":
        url = "https://www.wellcare.com/en/illinois/need-a-plan";
        break;
      default:
        alert("Please select an option before proceeding.");
        return;
    }
    window.location.href = url;
  };

  return (
    <div className="flex flex-col items-center justify-center p-6 space-y-4">
      <h2 className="text-2xl font-bold text-center">Ready to protect what matters most?</h2>
      <p className="text-muted-foreground text-center max-w-md">
        Get comprehensive coverage at competitive rates with our trusted insurance partners.
      </p>
      
      {/* Dropdown for selecting options */}
      <select
        className="mt-4 p-2 border border-gray-300 rounded"
        value={selectedOption}
        onChange={(e) => setSelectedOption(e.target.value)}
      >
        <option value="">Select an option</option>
        <option value="Aetna Medical Insurance">Aetna Medical Insurance</option>
        <option value="Blue Cross and Blue Shield">Blue Cross and Blue Shield</option>
        <option value="Humana Medical Insurance">Humana Medical Insurance</option>
        <option value="Cigna">Cigna</option>
        <option value="AARP">AARP</option>
        <option value="Medica">Medica</option>
        <option value="Wellcare">Wellcare</option>
      </select>

      {/* Redirect button */}
      <button
        onClick={handleBuyInsurance}
        className="mt-4 text-lg px-8 py-6 h-auto font-semibold bg-black text-white rounded-full"
      >
        Buy Insurance Now <ArrowRight className="ml-2 h-5 w-5" />
      </button>
    </div>
  );
}

