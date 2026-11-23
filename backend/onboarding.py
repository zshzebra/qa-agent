from datetime import date

from pydantic import BaseModel, Field, HttpUrl, EmailStr


class ExampleCompanyOnboarding(BaseModel):
    """Company information required for the current example onboarding flow."""

    name: str = Field(description="The name the company operates under.")
    website: HttpUrl = Field(description="The company's primary website.")
    industry: str = Field(description="The industry the company operates in.")
    abn: str = Field(description="The company's Australian Business Number.")
    address: str = Field(description="The company's primary business address.")
    phone: str = Field(description="The company's primary contact phone number.")
    email: EmailStr = Field(description="The company's primary contact email address.")
    date_first_active: date = Field(
        description="The date the company first became active."
    )
    founders: list[str] = Field(min_length=1, description="The name of the company's founder, as full names.")
    customer_samples: list[str] = Field(description="A few named examples of the business's customers, with their description.")
