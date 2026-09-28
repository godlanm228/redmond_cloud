#!/usr/bin/env bash
# One-time IAM setup for the A1 hunter. Run in OCI Cloud Shell by the account
# owner: it changes who may do what in the tenancy, so it is never run by code.
#
# Grants exactly one instance (the hub VM, via instance principal) the right to
# launch instances and read capacity reports. It cannot terminate instances,
# and no API key is stored on the VM. Safe to run again: an existing group is
# kept and an existing policy gets the current statements.
set -euo pipefail

: "${HUB_INSTANCE_ID:?set HUB_INSTANCE_ID to the hub VM OCID}"
TENANCY="${OCI_TENANCY:?run this in OCI Cloud Shell}"
NAME=redmond-a1-hunter

STATEMENTS="[
  \"Allow dynamic-group ${NAME} to manage instance-family in tenancy where request.operation != 'TerminateInstance'\",
  \"Allow dynamic-group ${NAME} to use volume-family in tenancy\",
  \"Allow dynamic-group ${NAME} to use virtual-network-family in tenancy\",
  \"Allow dynamic-group ${NAME} to manage compute-capacity-reports in tenancy\"
]"

GROUP_ID=$(oci iam dynamic-group list --all \
  --query "data[?name=='${NAME}'].id | [0]" --raw-output 2>/dev/null || true)
if [ -z "$GROUP_ID" ] || [ "$GROUP_ID" = "null" ]; then
  oci iam dynamic-group create --name "$NAME" \
    --description "Redmond hub VM: may launch the A1 instance" \
    --matching-rule "instance.id = '${HUB_INSTANCE_ID}'"
else
  echo "Dynamic group ${NAME} exists: ${GROUP_ID}"
fi

POLICY_ID=$(oci iam policy list --compartment-id "$TENANCY" --all \
  --query "data[?name=='${NAME}'].id | [0]" --raw-output 2>/dev/null || true)
if [ -z "$POLICY_ID" ] || [ "$POLICY_ID" = "null" ]; then
  oci iam policy create --compartment-id "$TENANCY" --name "$NAME" \
    --description "A1 hunter: launch instances, never terminate" \
    --statements "$STATEMENTS"
else
  oci iam policy update --policy-id "$POLICY_ID" --statements "$STATEMENTS" \
    --version-date "" --force
fi

echo "Done. Permissions can take a minute to apply."
