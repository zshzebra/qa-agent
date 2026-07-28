from datetime import date

from pydantic import BaseModel, Field, HttpUrl


class ExampleCompanyOnboarding(BaseModel):
    """Company information required for the current example onboarding flow."""

    name: str = Field(description="The name the company operates under.")
    website: HttpUrl = Field(description="The company's primary website.")
    industry: str = Field(description="The industry the company operates in.")
    abn: str = Field(description="The company's Australian Business Number.")
    address: str = Field(description="The company's primary business address.")
    phone: str = Field(description="The company's primary contact phone number.")
    email: str = Field(description="The company's primary contact email address.")
    date_first_active: date = Field(
        description="The date the company first became active."
    )
    founder: str = Field(description="The name of the company's founder.")
