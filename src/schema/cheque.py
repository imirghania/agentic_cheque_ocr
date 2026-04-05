from __future__ import annotations

from pydantic import BaseModel, Field


class BankInfo(BaseModel):
    phone: str | None = Field(default=None, description="Bank phone number")
    fax: str | None = Field(default=None, description="Bank fax number")
    website: str | None = Field(default=None, description="Bank website URL")
    email: str | None = Field(default=None, description="Bank email address")
    post_box: str | None = Field(default=None, description="Bank postal box number")


class ChequeData(BaseModel):
    bank_name: str | None = Field(default=None, description="Name of the bank")
    bank_branch: str | None = Field(default=None, description="Bank branch name or location")
    bank_info: BankInfo | None = Field(default=None, description="Bank contact details")
    cheque_number: str | None = Field(default=None, description="Cheque number")
    date: str | None = Field(default=None, description="Date written on the cheque")
    payee: str | None = Field(default=None, description="Name of the payee (who the cheque is made out to)")
    amount_in_words: str | None = Field(default=None, description="Amount written out in words")
    amount_in_numbers: str | None = Field(default=None, description="Amount in numeric form")
    payer: str | None = Field(default=None, description="Name of the payer (account holder who issued the cheque)")
    account_number: str | None = Field(default=None, description="Account number of the payer")
    micr_code: str | None = Field(default=None, description="MICR code (Magnetic Ink Character Recognition) at the bottom of the cheque")
    sort_code: str | None = Field(default=None, description="Bank sort code")
