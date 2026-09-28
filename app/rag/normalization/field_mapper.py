from __future__ import annotations

import logging
import re

from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FieldMappingResult:
    """
    Result of mapping source-specific field names
    to canonical real-estate field names.

    Attributes:
        mapped:
            Canonical field names and their values.

        unmapped:
            Source fields that could not be mapped.

        mapping_trace:
            Information about how each field was mapped.
            Useful for debugging and data lineage.
    """

    mapped: dict[str, Any] = field(default_factory=dict)
    unmapped: dict[str, Any] = field(default_factory=dict)
    mapping_trace: dict[str, str] = field(default_factory=dict)


class FieldMapper:
    """
    Map heterogeneous property field names to a canonical schema.

    Example:

        {
            "beds": 3,
            "baths": 2,
            "area_sqm": 1500,
            "property_category": "Apartment",
            "price_egp": 5000000
        }

    becomes:

        {
            "bedrooms": 3,
            "bathrooms": 2,
            "area": 1500,
            "property_type": "Apartment",
            "price": 5000000
        }

    Responsibilities:
        - Resolve field aliases.
        - Normalize field names.
        - Preserve unknown fields.
        - Keep mapping trace.

    Does NOT:
        - Extract values from free text.
        - Perform OCR.
        - Create embeddings.
        - Perform retrieval.
        - Guess missing values.
    """

    # ------------------------------------------------------------------
    # Canonical field aliases
    # ------------------------------------------------------------------

    FIELD_ALIASES: dict[str, set[str]] = {
        "property_id": {
            "property_id",
            "propertyid",
            "prop_id",
            "propid",
            "property_ref",
            "property_reference",
            "listing_id",
            "listingid",
            "unit_id",
            "unitid",
        },

        "property_type": {
            "property_type",
            "propertytype",
            "property_category",
            "propertycategory",
            "type",
            "unit_type",
            "unit_category",
            "category",
        },

        "transaction_type": {
            "transaction_type",
            "transactiontype",
            "listing_type",
            "listingtype",
            "deal_type",
            "dealtype",
            "purpose",
            "operation",
            "sale_or_rent",
            "sale_rent",
        },

        "price": {
            "price",
            "price_egp",
            "price_usd",
            "property_price",
            "propertyprice",
            "listing_price",
            "listingprice",
            "asking_price",
            "askingprice",
            "sale_price",
            "saleprice",
            "rent_price",
            "rentprice",
            "amount",
        },

        "currency": {
            "currency",
            "currency_code",
            "currencycode",
            "price_currency",
            "pricecurrency",
        },

        "area": {
            "area",
            "area_sqm",
            "area_sq_m",
            "area_m2",
            "property_area",
            "propertyarea",
            "unit_area",
            "unitarea",
            "size",
            "property_size",
            "propertysize",
            "sqft",
            "square_feet",
            "squarefeet",
            "sqm",
            "square_meters",
            "squaremeters",
            "built_up_area",
            "builtup_area",
            "bua",
        },

        "area_unit": {
            "area_unit",
            "areaunit",
            "unit",
            "measurement_unit",
            "size_unit",
        },

        "bedrooms": {
            "bedrooms",
            "bedroom",
            "beds",
            "bed",
            "num_bedrooms",
            "number_of_bedrooms",
            "number_bedrooms",
            "no_of_bedrooms",
            "no_bedrooms",
            "br",
        },

        "bathrooms": {
            "bathrooms",
            "bathroom",
            "baths",
            "bath",
            "num_bathrooms",
            "number_of_bathrooms",
            "number_bathrooms",
            "no_of_bathrooms",
            "no_bathrooms",
            "ba",
        },

        "location": {
            "location",
            "property_location",
            "propertylocation",
            "address",
            "property_address",
            "propertyaddress",
            "area_location",
            "location_area",
            "place",
        },

        "city": {
            "city",
            "city_name",
            "cityname",
            "location_governorate",
            "governorate",
            "governorate_name",
        },

        "district": {
            "district",
            "district_name",
            "districtname",
            "location_district",
            "area_name",
            "areaname",
        },

        "neighborhood": {
            "neighborhood",
            "neighbourhood",
            "neighborhood_name",
            "neighbourhood_name",
        },

        "compound": {
            "compound",
            "compound_name",
            "compoundname",
            "project",
            "project_name",
            "projectname",
            "development",
            "development_name",
        },

        "developer": {
            "developer",
            "developer_name",
            "developername",
            "developer_company",
            "developercompany",
        },

        "finishing": {
            "finishing",
            "finish",
            "finishing_type",
            "finishingtype",
            "condition",
            "property_condition",
            "propertycondition",
        },

        "delivery": {
            "delivery",
            "delivery_date",
            "deliverydate",
            "handover",
            "handover_date",
            "handoverdate",
            "completion",
            "completion_date",
        },

        "availability": {
            "availability",
            "available",
            "availability_status",
            "availabilitystatus",
            "unit_status",
            "status",
        },

        "floor": {
            "floor",
            "floor_number",
            "floornumber",
            "unit_floor",
            "unitfloor",
        },

        "total_floors": {
            "total_floors",
            "totalfloors",
            "number_of_floors",
            "num_floors",
            "floors_count",
        },

        "parking": {
            "parking",
            "parking_spaces",
            "parkingspaces",
            "garage",
            "garage_spaces",
            "garagespaces",
        },

        "amenities": {
            "amenities",
            "amenity",
            "features",
            "property_features",
            "propertyfeatures",
        },

        "description": {
            "description",
            "property_description",
            "propertydescription",
            "details",
            "property_details",
            "propertydetails",
            "listing_description",
        },

        "latitude": {
            "latitude",
            "lat",
            "property_latitude",
            "propertylatitude",
        },

        "longitude": {
            "longitude",
            "lng",
            "lon",
            "long",
            "property_longitude",
            "propertylongitude",
        },

        "source": {
            "source",
            "data_source",
            "datasource",
            "provider",
            "source_name",
        },

        "source_id": {
            "source_id",
            "sourceid",
            "external_id",
            "externalid",
            "external_reference",
            "externalreference",
        },
    }

    def __init__(
        self,
        *,
        preserve_unknown: bool = True,
    ) -> None:
        self.preserve_unknown = preserve_unknown

        self._alias_to_canonical = (
            self._build_alias_index()
        )

        logger.info(
            "FieldMapper initialized | "
            "aliases=%d | preserve_unknown=%s",
            len(self._alias_to_canonical),
            self.preserve_unknown,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def map_fields(
        self,
        fields: dict[str, Any],
    ) -> FieldMappingResult:
        """
        Map source fields to canonical field names.

        Example:

            {
                "Beds": 3,
                "Baths": 2,
                "Area_sqm": 1500,
                "Price_EGP": 5000000
            }

        becomes:

            {
                "bedrooms": 3,
                "bathrooms": 2,
                "area": 1500,
                "price": 5000000
            }
        """

        if not fields:
            logger.warning(
                "Field mapping skipped: empty input"
            )
            return FieldMappingResult()

        mapped: dict[str, Any] = {}
        unmapped: dict[str, Any] = {}
        mapping_trace: dict[str, str] = {}

        for source_field, value in fields.items():
            canonical_field = self.map_field_name(
                source_field
            )

            if canonical_field is None:
                if self.preserve_unknown:
                    unmapped[source_field] = value

                logger.debug(
                    "Field not mapped | source_field=%s",
                    source_field,
                )
                continue

            # Avoid silently overwriting a field when
            # multiple source fields map to the same
            # canonical field.
            if canonical_field in mapped:
                logger.warning(
                    "Duplicate canonical field detected | "
                    "canonical=%s | source=%s",
                    canonical_field,
                    source_field,
                )

                if self.preserve_unknown:
                    unmapped[source_field] = value

                continue

            mapped[canonical_field] = value
            mapping_trace[source_field] = canonical_field

        # --------------------------------------------------------------
        # Deterministic location fallback
        # --------------------------------------------------------------
        #
        # Some property datasets provide:
        #
        #   location_area = null
        #   location_district = "Sheikh Zayed"
        #
        # In that case, there is no direct location value.
        # Using the district as the broader location is a deterministic
        # fallback based on an existing canonical field.
        #
        if (
            mapped.get("location") is None
            and mapped.get("district")
        ):
            mapped["location"] = mapped["district"]
            mapping_trace["district"] = "location_fallback"

        logger.info(
            "Field mapping completed | "
            "input=%d | mapped=%d | unmapped=%d",
            len(fields),
            len(mapped),
            len(unmapped),
        )

        return FieldMappingResult(
            mapped=mapped,
            unmapped=unmapped,
            mapping_trace=mapping_trace,
        )

    def map_field_name(
        self,
        field_name: str,
    ) -> str | None:
        """
        Map one source field name to its canonical name.

        Returns:
            Canonical field name or None.
        """

        if not field_name:
            return None

        normalized_name = (
            self._normalize_field_name(field_name)
        )

        return self._alias_to_canonical.get(
            normalized_name
        )

    def map_to_dict(
        self,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return only the canonical mapped fields.

        Useful when the caller does not need mapping
        diagnostics.
        """

        result = self.map_fields(fields)

        return result.mapped

    def canonical_fields(self) -> list[str]:
        """
        Return all supported canonical field names.
        """

        return sorted(
            self.FIELD_ALIASES.keys()
        )

    def is_supported_field(
        self,
        field_name: str,
    ) -> bool:
        """
        Check whether a source field can be mapped.
        """

        return (
            self.map_field_name(field_name)
            is not None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_alias_index(
        self,
    ) -> dict[str, str]:
        """
        Build:

            normalized alias -> canonical field
        """

        alias_index: dict[str, str] = {}

        for canonical_field, aliases in (
            self.FIELD_ALIASES.items()
        ):
            # Canonical field itself is always accepted.
            all_aliases = set(aliases)
            all_aliases.add(canonical_field)

            for alias in all_aliases:
                normalized_alias = (
                    self._normalize_field_name(alias)
                )

                existing = alias_index.get(
                    normalized_alias
                )

                if (
                    existing is not None
                    and existing != canonical_field
                ):
                    raise ValueError(
                        "Field alias collision detected: "
                        f"'{alias}' maps to both "
                        f"'{existing}' and "
                        f"'{canonical_field}'"
                    )

                alias_index[
                    normalized_alias
                ] = canonical_field

        return alias_index

    @staticmethod
    def _normalize_field_name(
        field_name: str,
    ) -> str:
        """
        Normalize a field name before alias lookup.

        Examples:

            "Property Type"
                -> "property_type"

            "property-type"
                -> "property_type"

            " BEDROOMS "
                -> "bedrooms"
        """

        value = str(field_name).strip().lower()

        # Handle common separators.
        value = re.sub(
            r"[\s\-/]+",
            "_",
            value,
        )

        # Keep only letters, numbers and underscores.
        value = re.sub(
            r"[^a-z0-9_]",
            "",
            value,
        )

        # Collapse duplicate underscores.
        value = re.sub(
            r"_+",
            "_",
            value,
        )

        return value.strip("_")
