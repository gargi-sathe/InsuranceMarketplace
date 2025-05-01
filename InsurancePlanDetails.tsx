"use client";

import { useEffect, useState } from "react";

interface InsurancePlanDetailsProps {
  selectedPlan: string;
}

export default function InsurancePlanDetails({ selectedPlan }: InsurancePlanDetailsProps) {
  const [planDetails, setPlanDetails] = useState<string>("");

  const additionalDetails: { [key: string]: string } = {
    "Humana Group Medicare Advantage PPO Plan": `
      <div class="plan-container">
        <h3 class="font-bold text-xl">Plan Overview:</h3>
        <ul>
          <li><strong>Medical Coverage (Part A & B):</strong> Office visits, urgent care, emergency services, lab and radiology, ambulance.</li>
          <li><strong>Prescription Drug Coverage (Part D):</strong> No deductible; see stages in Chapter 6.</li>
          <li><strong>Supplemental Benefits:</strong> Telehealth, wellness programs, health coaching, home care.</li>
        </ul>
        
        <h3 class="font-bold text-xl mt-4">Cost Structure & Out-of-Pocket Maximum:</h3>
        <p><strong>Combined Deductible:</strong> $150</p>
        <p><strong>Combined Maximum Out-of-Pocket:</strong> $3,400</p>
        
        <h3 class="font-bold text-xl mt-4">Medical Services Coverage Highlights:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Service</th>
              <th class="border p-2">In-Network Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">Primary Care Visit</td>
              <td class="border p-2">$15 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Specialist Visit</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Emergency Room</td>
              <td class="border p-2">$50 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Urgent Care</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Ambulance Services</td>
              <td class="border p-2">$75 copay</td>
            </tr>
          </tbody>
        </table>
        
        <h3 class="font-bold text-xl mt-4">Prescription Drug Coverage Tiers:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Tier</th>
              <th class="border p-2">Description</th>
              <th class="border p-2">Copay/Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">1</td>
              <td class="border p-2">Preferred Generic</td>
              <td class="border p-2">$10</td>
            </tr>
            <tr>
              <td class="border p-2">2</td>
              <td class="border p-2">Generic</td>
              <td class="border p-2">$30</td>
            </tr>
            <tr>
              <td class="border p-2">3</td>
              <td class="border p-2">Preferred Brand</td>
              <td class="border p-2">25% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">4</td>
              <td class="border p-2">Non-Preferred Drug</td>
              <td class="border p-2">50% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">5</td>
              <td class="border p-2">Specialty</td>
              <td class="border p-2">33% coinsurance</td>
            </tr>
          </tbody>
        </table>
      </div>
    `,
    "Aetna Medicare Core (PPO) Plan": `
      <div class="plan-container">
        <h3 class="font-bold text-xl">Plan Overview:</h3>
        <ul>
          <li><strong>Medical Coverage:</strong> All Part A & B services per Medical Benefits Chart.</li>
          <li><strong>Prescription Drug Coverage (Part D):</strong> Deductible, Initial, and Catastrophic stages.</li>
          <li><strong>Supplemental Benefits:</strong> SilverSneakers, telehealth, care coordination.</li>
        </ul>
        
        <h3 class="font-bold text-xl mt-4">Cost Structure & Out-of-Pocket Maximum:</h3>
        <p><strong>Combined Deductible:</strong> $150</p>
        <p><strong>In-Network MOOP:</strong> $4,500</p>
        <p><strong>Combined MOOP:</strong> $8,900</p>
        
        <h3 class="font-bold text-xl mt-4">Medical Services Coverage Highlights:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Service</th>
              <th class="border p-2">In-Network Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">Primary Care Visit</td>
              <td class="border p-2">$15 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Specialist Visit</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Emergency Room</td>
              <td class="border p-2">$50 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Urgent Care</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Ambulance</td>
              <td class="border p-2">$75 copay</td>
            </tr>
          </tbody>
        </table>
        
        <h3 class="font-bold text-xl mt-4">Prescription Drug Coverage Tiers:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Tier</th>
              <th class="border p-2">Description</th>
              <th class="border p-2">Copay/Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">1</td>
              <td class="border p-2">Preferred Generic</td>
              <td class="border p-2">$10</td>
            </tr>
            <tr>
              <td class="border p-2">2</td>
              <td class="border p-2">Generic</td>
              <td class="border p-2">$30</td>
            </tr>
            <tr>
              <td class="border p-2">3</td>
              <td class="border p-2">Preferred Brand</td>
              <td class="border p-2">$60</td>
            </tr>
            <tr>
              <td class="border p-2">4</td>
              <td class="border p-2">Non-Preferred Drug</td>
              <td class="border p-2">25% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">5</td>
              <td class="border p-2">Specialty</td>
              <td class="border p-2">25% coinsurance</td>
            </tr>
          </tbody>
        </table>
      </div>
    `,
    "Cigna Preferred Medicare (HMO)": `
      <div class="plan-container">
        <h3 class="font-bold text-xl">Plan Overview:</h3>
        <ul>
          <li><strong>Medical Coverage (Part A & B):</strong> No deductible; copays apply.</li>
          <li><strong>Prescription Drug Coverage (Part D):</strong> No deductible; Part D stages apply.</li>
          <li><strong>Supplemental Benefits:</strong> ADAP, telehealth, extra support programs.</li>
        </ul>
        
        <h3 class="font-bold text-xl mt-4">Cost Structure & Out-of-Pocket Maximum:</h3>
        <p><strong>No Medical Deductible</strong></p>
        <p><strong>Medical MOOP:</strong> $4,200</p>
        
        <h3 class="font-bold text-xl mt-4">Medical Services Coverage Highlights:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Service</th>
              <th class="border p-2">In-Network Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">PCP Visit</td>
              <td class="border p-2">$0 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Specialist Visit</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Emergency Room</td>
              <td class="border p-2">$125 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Urgent Care</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Inpatient Hospital</td>
              <td class="border p-2">20% coinsurance</td>
            </tr>
          </tbody>
        </table>
        
        <h3 class="font-bold text-xl mt-4">Prescription Drug Coverage Tiers:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Tier</th>
              <th class="border p-2">Description</th>
              <th class="border p-2">Copay/Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">1</td>
              <td class="border p-2">Preferred Generic</td>
              <td class="border p-2">$10</td>
            </tr>
            <tr>
              <td class="border p-2">2</td>
              <td class="border p-2">Generic</td>
              <td class="border p-2">$30</td>
            </tr>
            <tr>
              <td class="border p-2">3</td>
              <td class="border p-2">Preferred Brand</td>
              <td class="border p-2">$47</td>
            </tr>
            <tr>
              <td class="border p-2">4</td>
              <td class="border p-2">Non-Preferred Drug</td>
              <td class="border p-2">$100</td>
            </tr>
            <tr>
              <td class="border p-2">5</td>
              <td class="border p-2">Specialty</td>
              <td class="border p-2">33% coinsurance</td>
            </tr>
          </tbody>
        </table>
      </div>
    `,
    "AARP Medicare Advantage from UHC (PPO)": `
      <div class="plan-container">
        <h3 class="font-bold text-xl">Plan Overview:</h3>
        <ul>
          <li><strong>Medical Coverage (Part A & B):</strong> Standard copays and coinsurance.</li>
          <li><strong>Drug Coverage (Part D):</strong> No deductible; Part D stages apply.</li>
          <li><strong>Supplemental Benefits:</strong> Telehealth, UCard benefits.</li>
        </ul>
        
        <h3 class="font-bold text-xl mt-4">Cost Structure & Out-of-Pocket Maximum:</h3>
        <p><strong>Combined Deductible:</strong> $150</p>
        <p><strong>In-Network MOOP:</strong> $4,500</p>
        
        <h3 class="font-bold text-xl mt-4">Medical Services Coverage Highlights:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Service</th>
              <th class="border p-2">In-Network Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">Primary Care Visit</td>
              <td class="border p-2">$0 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Specialist Visit</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Emergency Room</td>
              <td class="border p-2">$50 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Urgent Care</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Inpatient Hospital</td>
              <td class="border p-2">$395 copay/stay</td>
            </tr>
          </tbody>
        </table>
        
        <h3 class="font-bold text-xl mt-4">Prescription Drug Coverage Tiers:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Tier</th>
              <th class="border p-2">Description</th>
              <th class="border p-2">Copay/Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">1</td>
              <td class="border p-2">Preferred Generic</td>
              <td class="border p-2">$0</td>
            </tr>
            <tr>
              <td class="border p-2">2</td>
              <td class="border p-2">Generic</td>
              <td class="border p-2">$10</td>
            </tr>
            <tr>
              <td class="border p-2">3</td>
              <td class="border p-2">Preferred Brand</td>
              <td class="border p-2">$47</td>
            </tr>
            <tr>
              <td class="border p-2">4</td>
              <td class="border p-2">Non-Preferred Drug</td>
              <td class="border p-2">$100</td>
            </tr>
            <tr>
              <td class="border p-2">5</td>
              <td class="border p-2">Specialty</td>
              <td class="border p-2">33% coinsurance</td>
            </tr>
          </tbody>
        </table>
      </div>
    `,
    "Medica Group Advantage Solution (PPO)": `
      <div class="plan-container">
        <h3 class="font-bold text-xl">Plan Overview:</h3>
        <ul>
          <li><strong>Medical Coverage:</strong> Part A & B services; no referrals required.</li>
          <li><strong>Prescription Drug Coverage:</strong> Part D stages; no deductible.</li>
          <li><strong>Supplemental Benefits:</strong> ADAP, telehealth.</li>
        </ul>
        
        <h3 class="font-bold text-xl mt-4">Cost Structure & Out-of-Pocket Maximum:</h3>
        <p><strong>Medical Deductible:</strong> $0</p>
        <p><strong>In-Network MOOP (Part A/B):</strong> $4,200</p>
        
        <h3 class="font-bold text-xl mt-4">Medical Services Coverage Highlights:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Service</th>
              <th class="border p-2">In-Network Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">PCP Visit</td>
              <td class="border p-2">$0 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Specialist Visit</td>
              <td class="border p-2">$30 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Emergency Room</td>
              <td class="border p-2">$140 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Urgent Care</td>
              <td class="border p-2">$60 copay</td>
            </tr>
          </tbody>
        </table>
        
        <h3 class="font-bold text-xl mt-4">Prescription Drug Coverage Tiers:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Tier</th>
              <th class="border p-2">Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">1</td>
              <td class="border p-2">$5</td>
            </tr>
            <tr>
              <td class="border p-2">2</td>
              <td class="border p-2">$10</td>
            </tr>
            <tr>
              <td class="border p-2">3</td>
              <td class="border p-2">25% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">4</td>
              <td class="border p-2">44% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">5</td>
              <td class="border p-2">28% coinsurance</td>
            </tr>
          </tbody>
        </table>
      </div>
    `,
    "Wellcare Premium Ultra Open (PPO)": `
      <div class="plan-container">
        <h3 class="font-bold text-xl">Plan Overview:</h3>
        <ul>
          <li><strong>Medical Coverage:</strong> Part A & B services; no referrals required.</li>
          <li><strong>Prescription Drug Coverage:</strong> Part D stages; no deductible.</li>
          <li><strong>Supplemental Benefits:</strong> ADAP, telehealth.</li>
        </ul>
        
        <h3 class="font-bold text-xl mt-4">Cost Structure & Out-of-Pocket Maximum:</h3>
        <p><strong>Medical Deductible:</strong> $0</p>
        <p><strong>In-Network MOOP (Part A/B):</strong> $4,200</p>
        
        <h3 class="font-bold text-xl mt-4">Medical Services Coverage Highlights:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Service</th>
              <th class="border p-2">In-Network Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">PCP Visit</td>
              <td class="border p-2">$0 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Specialist Visit</td>
              <td class="border p-2">$25 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Emergency Room</td>
              <td class="border p-2">$140 copay</td>
            </tr>
            <tr>
              <td class="border p-2">Urgent Care</td>
              <td class="border p-2">$60 copay</td>
            </tr>
          </tbody>
        </table>
        
        <h3 class="font-bold text-xl mt-4">Prescription Drug Coverage Tiers:</h3>
        <table class="table-auto w-full text-left border-collapse mt-2">
          <thead>
            <tr>
              <th class="border p-2">Tier</th>
              <th class="border p-2">Cost</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="border p-2">1</td>
              <td class="border p-2">$5</td>
            </tr>
            <tr>
              <td class="border p-2">2</td>
              <td class="border p-2">$10</td>
            </tr>
            <tr>
              <td class="border p-2">3</td>
              <td class="border p-2">25% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">4</td>
              <td class="border p-2">44% coinsurance</td>
            </tr>
            <tr>
              <td class="border p-2">5</td>
              <td class="border p-2">28% coinsurance</td>
            </tr>
          </tbody>
        </table>
      </div>
    `,
  };

  useEffect(() => {
    setPlanDetails(additionalDetails[selectedPlan] || "");
  }, [selectedPlan]);

  return (
    <div className="bg-white p-4 rounded">
      <div dangerouslySetInnerHTML={{ __html: planDetails }} />
    </div>
  );
}
