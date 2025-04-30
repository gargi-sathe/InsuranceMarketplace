"use client"

import { useState } from "react";
import { ArrowRight } from "lucide-react";

export default function BuyInsuranceButton() {
  const [selectedOption, setSelectedOption] = useState<string>("");
  const [documentsRequired, setDocumentsRequired] = useState<string[]>([]);

  // Mapping insurance options to document requirements
  const insuranceDocuments: { [key: string]: string[] } = {
    "Aetna Medical Insurance": [
      "Proof of Identity (Driver's License, Passport)",
      "Proof of Citizenship or Legal Residency",
      "Income Verification (Pay stubs, Tax returns)",
      "Proof of Address (Utility bill, Lease agreement)",
      "Health Information (Medical history, prescriptions)"
    ],
    "Blue Cross and Blue Shield": [
      "Proof of Identity",
      "Proof of Citizenship or Legal Residency",
      "Income Verification",
      "Proof of Address",
      "Current Health Insurance Information"
    ],
    "Humana Medical Insurance": [
      "Proof of Identity",
      "Proof of Citizenship or Legal Residency",
      "Income Verification",
      "Proof of Address",
      "Health History"
    ],
    "Cigna": [
      "Proof of Identity",
      "Proof of Citizenship or Legal Residency",
      "Income Verification",
      "Health Information",
      "Proof of Address"
    ],
    "AARP": [
      "Proof of Identity",
      "Proof of Age",
      "Proof of Citizenship or Legal Residency",
      "Income Verification",
      "Proof of Address"
    ],
    "Medica": [
      "Proof of Identity",
      "Proof of Citizenship or Legal Residency",
      "Income Verification",
      "Proof of Address",
      "Health Information"
    ],
    "WellCare": [
      "Proof of Identity",
      "Proof of Citizenship or Legal Residency",
      "Income Verification",
      "Proof of Address"
    ]
  };

  const handleBuyInsurance = () => {
    let url = "";
    switch (selectedOption) {
      case "Aetna Medical Insurance":
        url = "https://enrollmedicare.aetna.com/s/shop?tfn=&ZipCode=60607&CountyFIPS=17031&PlanYear=2025&step=PlanList";
        break;
      case "Blue Cross and Blue Shield":
        url = "https://www.bcbsil.com/medicare";
        break;
      case "Humana Medical Insurance":
        url = "https://shop.humana-medicareadvantage.com/?pspt=4ce0d070-2601-11f0-b20c-0db53d0967c7";
        break;
      case "Cigna":
        url = "https://plans.cigna.com/?zip=60608&fip=17031&PlanType=MAPD";
        break;
      case "AARP":
        url = "https://www.aarpmedicareplans.com/health-plans/plan-summary/60608/031/2025#MA";
        break;
      case "Medica":
        url = "https://medica.isf.io/2025/g/7876cf047cb74e33aaecd5e44ff17615/AssistedShopping?step=3";
        break;
      case "WellCare":
        url = "https://www.wellcare.com/en/illinois/need-a-plan";
        break;
      default:
        alert("Please select an option before proceeding.");
        return;
    }
    window.location.href = url;
  };

  // Update documents list when an option is selected
  const handleOptionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selected = e.target.value;
    setSelectedOption(selected);
    setDocumentsRequired(insuranceDocuments[selected] || []);
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
        onChange={handleOptionChange}
      >
        <option value="">Select an option</option>
        <option value="Aetna Medical Insurance">Aetna Medical Insurance</option>
        <option value="Blue Cross and Blue Shield">Blue Cross and Blue Shield</option>
        <option value="Humana Medical Insurance">Humana Medical Insurance</option>
        <option value="Cigna">Cigna</option>
        <option value="AARP">AARP</option>
        <option value="Medica">Medica</option>
        <option value="WellCare">WellCare</option>
      </select>

      {/* Displaying documents required based on the selected insurance */}
      {selectedOption && (
        <div className="mt-6">
          <h3 className="text-xl font-semibold">Documents Required:</h3>
          <ul className="list-disc pl-6">
            {documentsRequired.map((doc, index) => (
              <li key={index} className="text-muted-foreground">{doc}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Redirect button */}
      <button
        onClick={handleBuyInsurance}
        className="mt-4 text-lg px-8 py-6 h-auto font-semibold bg-black text-white rounded-full flex items-center justify-center space-x-2"
      >
        Buy Insurance Now <ArrowRight className="ml-2 h-5 w-5" />
      </button>
    </div>
  );
}
